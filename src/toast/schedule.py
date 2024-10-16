# Copyright (c) 2019-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys
from datetime import datetime

import dateutil
import dateutil.parser
import ephem
import numpy as np
from astropy import units as u
from astropy.table import Column, QTable

from .coordinates import to_DJD
from .timing import Timer, function_timer
from .utils import Environment, Logger


class Scan(object):
    """Base class for simulated telescope scan properties for one observation.

    We use python datetime for specifying times.  These are trivially convertable to
    astrometry packages.

    Args:
        name (str):  Arbitrary name (does not have to be unique).
        start (datetime):  The start time of the scan.
        stop (datetime):  The stop time of the scan.
    """

    def __init__(self, name=None, start=None, stop=None):
        self.name = name
        if start is None:
            raise RuntimeError("you must specify the start time")
        if stop is None:
            raise RuntimeError("you must specify the stop time")
        self.start = start
        self.stop = stop


class GroundScan(Scan):
    """Simulated ground telescope scan properties for one observation.

    Args:
        name (str):  Arbitrary name (does not have to be unique).
        start (datetime):  The start time of the scan.
        stop (datetime):  The stop time of the scan.
        boresight_angle (Quantity):  Boresight rotation angle.
        az_min (Quantity):  The minimum Azimuth value of each sweep.
        az_max (Quantity):  The maximum Azimuth value of each sweep.
        el (Quantity):  The nominal Elevation of the scan.
        scan_indx (int):  The current pass of this patch in the overall schedule.
        subscan_indx (int):  The current sub-pass of this patch in the overall schedule.

    """

    def __init__(
        self,
        name=None,
        start=None,
        stop=None,
        boresight_angle=0 * u.degree,
        az_min=0 * u.degree,
        az_max=0 * u.degree,
        el=0 * u.degree,
        scan_indx=0,
        subscan_indx=0,
    ):
        super().__init__(name=name, start=start, stop=stop)
        self.boresight_angle = boresight_angle
        self.az_min = az_min
        self.az_max = az_max
        self.el = el
        self.rising = az_min.to_value(u.degree) % 360 < 180
        self.scan_indx = scan_indx
        self.subscan_indx = subscan_indx
        self.ra_min = None
        self.ra_max = None
        self.ra_mean = None
        self.dec_min = None
        self.dec_max = None
        self.dec_mean = None

    def __repr__(self):
        start = self.start.isoformat(timespec="seconds")
        val = f"<GroundScan '{self.name}' "
        val += f"at {start} with El = {self.el}, Az {self.az_min} -- {self.az_max}, "
        val += f"RA = {self.ra_min:.1f} < {self.ra_mean:.1f} < {self.ra_max:.1f}, "
        val += f"Dec {self.dec_min:.1f} < {self.dec_mean:.1f} < {self.dec_max:.1f}>"
        return val

    @function_timer
    def min_sso_dist(self, sso_az_begin, sso_el_begin, sso_az_end, sso_el_end):
        """Rough minimum angle between the boresight and a solar system object.

        Args:
            sso_az_begin (Quantity):  Object starting Azimuth
            sso_el_begin (Quantity):  Object starting Elevation
            sso_az_end (Quantity):  Object final Azimuth
            sso_el_end (Quantity):  Object final Elevation

        Returns:
            (Quantity):  The minimum angle.

        """
        sso_vec1 = hp.dir2vec(
            sso_az_begin.to_value(u.degree),
            sso_el_begin.to_value(u.degree),
            lonlat=True,
        )
        sso_vec2 = hp.dir2vec(
            sso_az_end.to_value(u.degree), sso_el_end.to_value(u.degree), lonlat=True
        )
        az1 = self.az_min.to_value(u.degree)
        az2 = self.az_max.to_value(u.degree)
        if az2 < az1:
            az2 += 360.0
        n = 100
        az = np.linspace(az1, az2, n)
        el = np.ones(n) * self.el.to_value(u.degree)
        vec = hp.dir2vec(az, el, lonlat=True)
        dist1 = np.degrees(np.arccos(np.dot(sso_vec1, vec)))
        dist2 = np.degrees(np.arccos(np.dot(sso_vec2, vec)))
        result = min(np.amin(dist1), np.amin(dist2))
        return result * u.degree

    @function_timer
    def get_extent(self, observer):
        """Calculate the boresight scan range in Celestial coordinates
        based on time, azimuth and elevation
        """
        # Time grid
        t_step = 900.0  # 15 minutes in seconds
        start = self.start.timestamp()
        stop = self.stop.timestamp()
        delta_t = stop - start
        nstep_t = max(3, int(delta_t / t_step))
        times = np.linspace(self.start.timestamp(), self.stop.timestamp(), nstep_t)
        # Az grid
        az_step = np.radians(10)  # 10 deg in radians
        az_min = self.az_min.to_value(u.rad)
        az_max = self.az_max.to_value(u.rad)
        delta_az = az_max - az_min
        nstep_az = max(3, int(delta_az / az_step))
        azs = np.linspace(az_min, az_max, nstep_az)
        # Elevation
        el = self.el.to_value(u.rad)
        # Evaluate
        ras, decs = [], []
        for t in times:
            observer.date = to_DJD(t)
            for az in azs:
                ra, dec = np.degrees(observer.radec_of(az, el))
                ras.append(ra)
                decs.append(dec)
        ras = np.unwrap(ras, period=360)
        decs = np.array(decs)
        if np.mean(ras) < 0:
            ras += 360
        elif np.mean(ras) > 360:
            ras -= 360
        self.ra_min = np.amin(ras) * u.deg
        self.ra_max = np.amax(ras) * u.deg
        self.ra_mean = np.mean(ras) * u.deg
        self.dec_min = np.amin(decs) * u.deg
        self.dec_max = np.amax(decs) * u.deg
        self.dec_mean = np.mean(decs) * u.deg
        return


class SatelliteScan(Scan):
    """Simulated satellite telescope scan properties for one observation.

    This class assumes a simplistic model where the nominal precession axis is pointing
    in the anti-sun direction (from a location such as at L2).  This class just
    specifies the rotation rates about this axis and also about the spin axis.  The
    opening angles are part of the Telescope and not specified here.

    Args:
        name (str):  Arbitrary name (does not have to be unique).
        start (datetime):  The start time of the scan.
        stop (datetime):  The stop time of the scan.
        prec_period (Quantity):  The time for one revolution about the precession axis.
        spin_period (Quantity):  The time for one revolution about the spin axis.

    """

    def __init__(
        self,
        name=None,
        start=None,
        stop=None,
        prec_period=0 * u.minute,
        spin_period=0 * u.minute,
    ):
        super().__init__(name=name, start=start, stop=stop)
        self.prec_period = prec_period
        self.spin_period = spin_period

    def __repr__(self):
        val = "<SatelliteScan '{}' at {} with prec period {}, spin period {}>".format(
            self.name,
            self.start.isoformat(timespec="seconds"),
            self.prec_period,
            self.spin_period,
        )
        return val


class GroundSchedule(object):
    """Class representing a ground based observing schedule.

    A schedule is a collection of scans, with some extra methods for doing I/O.

    Args:
        scans (list):  A list of GroundScan instances or None.
        site_name (str):  The name of the site for this schedule.
        telescope_name (str):  The name of the telescope for this schedule.
        site_lat (Quantity):  The site latitude.
        site_lon (Quantity):  The site longitude.
        site_alt (Quantity):  The site altitude.
    """

    def __init__(
        self,
        scans=None,
        site_name="Unknown",
        telescope_name="Unknown",
        site_lat=0 * u.degree,
        site_lon=0 * u.degree,
        site_alt=0 * u.meter,
    ):
        self.scans = scans
        if scans is None:
            self.scans = list()
        else:
            for sc in self.scans:
                if not isinstance(sc, GroundScan):
                    raise RuntimeError("only GroundScan instances are supported")
        self.site_name = site_name
        self.telescope_name = telescope_name
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.site_alt = site_alt

    def __repr__(self):
        val = "<GroundSchedule "
        val += f"site={self.site_name} at "
        val += f"{self.site_lat}, {self.site_lon}, {self.site_alt} "
        val += f"telescope {self.telescope_name} "
        val += "with "
        if self.scans is None:
            val += "0 scans>"
            return val
        else:
            val += "{} scans".format(len(self.scans))
        if len(self.scans) < 5:
            for sc in self.scans:
                val += "\n  {}".format(sc)
        else:
            for sc in self.scans[:2]:
                val += "\n  {}".format(sc)
            val += "\n  ... "
            for sc in self.scans[-2:]:
                val += "\n  {}".format(sc)
        val += "\n>"
        return val

    @function_timer
    def read(
            self,
            schedule_file,
            file_split=None,
            comm=None,
            field_separator="|",
    ):
        """Load a ground observing schedule from a file.

        This loads scans from a file and appends them to the internal list of scans.
        The resulting combined scan list is optionally sorted.

        Args:
            schedule_file (str):  The file to load.
            file_split (tuple):  If not None, only use a subset of the schedule file.
                The arguments are (isplit, nsplit) and only observations that satisfy
                'scan index modulo nsplit == isplit' are included.
            comm (MPI.Comm):  Optional communicator to broadcast the schedule across.
            field_separator (str):  Field separator in the schedule file.  If the
                separator is not found in the string, use white space instead

        Returns:
            None

        """
        log = Logger.get()

        def _parse_line(line):
            """Parse one line of the schedule file"""
            if line.startswith("#"):
                return None
            fields = line.split(field_separator)
            if len(fields) == 1:
                # Failed ... try with white space
                fields = line.split()
            else:
                # Separating with anything but white space can
                # leave excess space
                fields = [field.strip() for field in fields]
            nfield = len(fields)
            if nfield == 9:
                # Concise schedule format with correct date/time parsing
                (
                    start_time,
                    stop_time,
                    boresight_angle,
                    name,
                    azmin,
                    azmax,
                    el,
                    scan,
                    subscan,
                ) = fields
            elif nfield == 11:
                # Concise schedule format is default after 2023-02-13
                (
                    start_date,
                    start_time,
                    stop_date,
                    stop_time,
                    boresight_angle,
                    name,
                    azmin,
                    azmax,
                    el,
                    scan,
                    subscan,
                ) = fields
                start_time = start_date + " " + start_time
                stop_time = stop_date + " " + stop_time
            elif nfield == 22:
                # Verbose format with correct date/time parsing
                (
                    start_time,
                    stop_time,
                    mjdstart,
                    mjdstop,
                    boresight_angle,
                    name,
                    azmin,
                    azmax,
                    el,
                    rs,
                    sun_el1,
                    sun_az1,
                    sun_el2,
                    sun_az2,
                    moon_el1,
                    moon_az1,
                    moon_el2,
                    moon_az2,
                    moon_phase,
                    scan,
                    subscan,
                    cumulative_fraction,
                ) = fields
            else:
                # Old (verbose) schedule
                (
                    start_date,
                    start_time,
                    stop_date,
                    stop_time,
                    mjdstart,
                    mjdstop,
                    boresight_angle,
                    name,
                    azmin,
                    azmax,
                    el,
                    rs,
                    sun_el1,
                    sun_az1,
                    sun_el2,
                    sun_az2,
                    moon_el1,
                    moon_az1,
                    moon_el2,
                    moon_az2,
                    moon_phase,
                    scan,
                    subscan,
                    cumulative_fraction,
                ) = fields
                start_time = start_date + " " + start_time
                stop_time = stop_date + " " + stop_time
            try:
                start_time = dateutil.parser.parse(start_time + " +0000")
                stop_time = dateutil.parser.parse(stop_time + " +0000")
            except Exception:
                start_time = dateutil.parser.parse(start_time)
                stop_time = dateutil.parser.parse(stop_time)
            return GroundScan(
                name,
                start_time,
                stop_time,
                float(boresight_angle) * u.degree,
                float(azmin) * u.degree,
                float(azmax) * u.degree,
                float(el) * u.degree,
                scan,
                subscan,
            )

        if comm is None or comm.rank == 0:
            log.info(f"Loading schedule from {schedule_file}")
            isplit = None
            nsplit = None
            if file_split is not None:
                isplit, nsplit = file_split
            scan_counters = dict()

            read_header = True
            last_name = None
            total_time = 0

            with open(schedule_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    if "SPECIAL" in line:
                        continue
                    if read_header:
                        fields = line.split(field_separator)
                        if len(fields) == 1:
                            # Failed ... try with white space
                            fields = line.split()
                        else:
                            # Separating with anything but white space can
                            # leave excess space
                            fields = [field.strip() for field in fields]
                        (
                            site_name,
                            telescope_name,
                            site_lat,
                            site_lon,
                            site_alt,
                        ) = fields
                        self.site_name = site_name
                        self.telescope_name = telescope_name
                        self.site_lat = float(site_lat) * u.degree
                        self.site_lon = float(site_lon) * u.degree
                        self.site_alt = float(site_alt) * u.meter
                        read_header = False
                        continue
                    gscan = _parse_line(line)
                    if nsplit is not None:
                        # Only accept 1 / `nsplit` of the rising and setting
                        # scans in patch `name`.  Selection is performed
                        # during the first subscan.
                        if name != last_name:
                            if name not in scan_counters:
                                scan_counters[name] = dict()
                            counter = scan_counters[name]
                            # Separate counters for rising and setting scans
                            ckey = "S"
                            if gscan.rising:
                                ckey = "R"
                            if ckey not in counter:
                                counter[ckey] = 0
                            else:
                                counter[ckey] += 1
                            iscan = counter[ckey]
                        last_name = name
                        if iscan % nsplit != isplit:
                            continue
                    total_time += (gscan.stop - gscan.start).total_seconds()
                    self.scans.append(gscan)
            if total_time > 2 * 86400:
                total_time = f"{total_time / 86400:.3f} days"
            elif total_time > 2 * 3600:
                total_time = f"{total_time / 3600:.3} hours"
            else:
                total_time = f"{total_time / 60:.3} minutes"
            log.info(
                f"Loaded {len(self.scans)} scans from {schedule_file} totaling {total_time}."
            )
        if comm is not None:
            self.site_name = comm.bcast(self.site_name, root=0)
            self.telescope_name = comm.bcast(self.telescope_name, root=0)
            self.site_lat = comm.bcast(self.site_lat, root=0)
            self.site_lon = comm.bcast(self.site_lon, root=0)
            self.site_alt = comm.bcast(self.site_alt, root=0)
            self.scans = comm.bcast(self.scans, root=0)

    @function_timer
    def sort_by_name(self):
        """Sort schedule by target name"""
        self.scans = sorted(self.scans, key=lambda scn: scn.name)

    def sort_by_RA(self):
        """Sort schedule by boresight RA"""
        observer = ephem.Observer()
        observer.lon = str(self.site_lon.to_value(u.deg))
        observer.lat = str(self.site_lat.to_value(u.deg))
        observer.elevation = self.site_alt.to_value(u.m)
        observer.epoch = "2000"
        observer.temp = 0  # Celsius
        observer.compute_pressure()
        for scan in self.scans:
            if scan.ra_mean is None:
                scan.get_extent(observer)
        self.scans = sorted(self.scans, key=lambda scn: scn.ra_mean)

    @function_timer
    def write(self, schedule_file):
        # FIXME:  We should have more robust format here (e.g. ECSV) and then use
        # This class when building the schedule as well.
        raise NotImplementedError("New ground schedule format not yet implemented")


class SatelliteSchedule(object):
    """Class representing a satellite observing schedule.

    A schedule is a collection of scans, with some extra methods for doing I/O.

    Args:
        scans (list):  A list of SatelliteScan instances or None.
        site_name (str):  The name of the site for this schedule.
        telescope_name (str):  The name of the telescope for this schedule.

    """

    def __init__(
        self,
        scans=None,
        site_name="Unknown",
        telescope_name="Unknown",
    ):
        self.scans = scans
        if scans is None:
            self.scans = list()
        else:
            for sc in self.scans:
                if not isinstance(sc, SatelliteScan):
                    raise RuntimeError("only SatelliteScan instances are supported")
        self.site_name = site_name
        self.telescope_name = telescope_name

    def __repr__(self):
        val = "<SatelliteSchedule "
        val += f"site={self.site_name} "
        val += f"telescope={self.telescope_name} "
        val += "with "
        if self.scans is None:
            val += "0 scans>"
            return val
        else:
            val += "{} scans".format(len(self.scans))
        if len(self.scans) < 5:
            for sc in self.scans:
                val += "\n  {}".format(sc)
        else:
            for sc in self.scans[:2]:
                val += "\n  {}".format(sc)
            val += "\n  ... "
            for sc in self.scans[-2:]:
                val += "\n  {}".format(sc)
        val += "\n>"
        return val

    @function_timer
    def read(self, schedule_file, comm=None, sort=False):
        """Load a satellite observing schedule from a file.

        This loads scans from a file and appends them to the internal list of scans.
        The resulting combined scan list is optionally sorted.

        Args:
            schedule_file (str):  The file to load.
            comm (MPI.Comm):  Optional communicator to broadcast the schedule across.
            sort (bool):  If True, sort the combined scan list by name.

        Returns:
            None

        """
        log = Logger.get()
        if comm is None or comm.rank == 0:
            log.info(f"Loading schedule from {schedule_file}")
            data = QTable.read(schedule_file, format="ascii.ecsv")
            self.telescope_name = data.meta["telescope_name"]
            self.site_name = data.meta["site_name"]
            for row in data:
                tstart = datetime.fromisoformat(row["start"])
                tstop = datetime.fromisoformat(row["stop"])
                self.scans.append(
                    SatelliteScan(
                        name=row["name"],
                        start=tstart,
                        stop=tstop,
                        prec_period=row["prec_period"],
                        spin_period=row["spin_period"],
                    )
                )
            if sort:
                sortedscans = sorted(self.scans, key=lambda scn: scn.name)
                self.scans = sortedscans
        if comm is not None:
            self.scans = comm.bcast(self.scans, root=0)

    @function_timer
    def write(self, schedule_file):
        """Write satellite schedule to a file.

        This writes the internal scan list to the specified file.

        Args:
            schedule_file (str):  The file to write.

        Returns:
            None

        """
        out = QTable(
            [
                Column(name="name", data=[x.name for x in self.scans]),
                Column(
                    name="start", data=[x.start.isoformat(sep="T") for x in self.scans]
                ),
                Column(
                    name="stop", data=[x.stop.isoformat(sep="T") for x in self.scans]
                ),
                Column(
                    name="prec_period",
                    data=[x.prec_period.to_value(u.minute) for x in self.scans],
                    unit=u.minute,
                ),
                Column(
                    name="spin_period",
                    data=[x.spin_period.to_value(u.minute) for x in self.scans],
                    unit=u.minute,
                ),
            ]
        )
        out.meta["telescope_name"] = self.telescope_name
        out.meta["site_name"] = self.site_name

        out.write(schedule_file, format="ascii.ecsv", overwrite=True)
