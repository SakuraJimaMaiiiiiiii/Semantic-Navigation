"""Planning-only vehicle columns for return; never modify mapping/localization."""
import copy
from dataclasses import is_dataclass, replace
import numpy as np

from mission.rfly_mission_controller import RflyMissionController
from planner.local_avoidance_planner import LocalPlan
from .vehicle_constraints import footprint_cells, expanded_boxes, segment_hits_vehicle


class VehiclePlanVeto:
    """Compose with the existing planner/guard, including its execution certificate."""
    def __init__(self, inner, objects, config):
        self.inner, self.objects, self.navigation_config = inner, objects, config

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def revalidate_plan(self, plan, current, *, certificate=None):
        checked = self.inner.revalidate_plan(plan, current, certificate=certificate)
        if checked.can_move:
            for identity,low,high in expanded_boxes(self.objects(), self.navigation_config):
                if segment_hits_vehicle(current, checked.local_target_ned, low, high):
                    self.inner._execution_rejection = dict(reason='vehicle_no_overflight',vehicle=identity)
                    return LocalPlan('semantic_vehicle_blocked', None, np.empty((0,3)), 0.)
        return checked


class VehicleMotionProxy:
    def __init__(self, vehicle, objects, config):
        self.vehicle, self.objects, self.config_for_constraints = vehicle, objects, config
        self._planner_source = self._planner_veto = None

    def __getattr__(self, name):
        return getattr(self.vehicle, name)

    def move_waypoint_guided(self, *args, **kwargs):
        planner = kwargs.get('local_planner')
        if planner is None:
            raise RuntimeError('Vehicle-aware motion requires an execution planner')
        # The flight controller keys its asynchronous worker by planner identity.
        # Rewrapping every one-second return horizon restarts that worker and
        # resets clearance settling with repeated waiting_for_plan intervals.
        if planner is not self._planner_source:
            self._planner_source = planner
            self._planner_veto = VehiclePlanVeto(planner, self.objects, self.config_for_constraints)
        kwargs['local_planner'] = self._planner_veto
        return self.vehicle.move_waypoint_guided(*args, **kwargs)


class VehiclePlanningMap:
    """A derived snapshot for return A*: observed XY footprints become columns.

    Raw occupied/free evidence, semantic labels and saved source maps stay intact.
    Normal NavigationGrid body clearance is applied by the original return code.
    """
    def __init__(self, source, objects, navigator):
        self.source,self.objects,self.navigator=source,objects,navigator
        self._key=self._snapshot=None

    def __getattr__(self,name):
        return getattr(self.source,name)

    @property
    def revision(self):
        geometry=tuple((o['id'],tuple(o['bbox_3d']['min']),tuple(o['bbox_3d']['max'])) for o in self.objects())
        return (getattr(self.source,'revision',0),geometry)

    def snapshot_with_revision(self):
        return self.snapshot(), self.revision

    def snapshot(self):
        raw=self.source.snapshot()
        objects=self.objects()
        geometry=tuple((o['id'],tuple(o['bbox_3d']['min']),tuple(o['bbox_3d']['max'])) for o in objects)
        key=(raw.timestamp,geometry,self.navigator.altitude)
        if key==self._key:
            return self._snapshot
        if not objects:
            return raw
        r=float(raw.resolution)
        footprints=set()
        for obj in objects:
            footprints.update(footprint_cells(obj['bbox_3d']['min'],obj['bbox_3d']['max'],r,np.zeros(2)))
        xy=np.asarray(sorted(footprints),dtype=np.int64).reshape(-1,2)
        altitude=self.navigator.altitude
        half=self.navigator.config.vertical_half_extent
        zlo=int(np.floor((altitude-half)/r));zhi=int(np.ceil((altitude+half)/r))
        if len(raw.indices):
            zlo=min(zlo,int(raw.indices[:,2].min()));zhi=max(zhi,int(raw.indices[:,2].max()))
        if len(xy)*(zhi-zlo+1)>1_000_000:
            raise RuntimeError('Vehicle return-column budget exceeded')
        zs=np.arange(zlo,zhi+1,dtype=np.int64)
        columns=np.column_stack((np.repeat(xy,len(zs),axis=0),np.tile(zs,len(xy))))
        keep=np.fromiter((tuple(cell) not in footprints for cell in raw.indices[:,:2]),dtype=bool,count=len(raw.indices))
        indices=np.concatenate((raw.indices[keep],columns))
        odds=np.concatenate((raw.log_odds[keep],np.full(len(columns),raw.occupied_threshold,dtype=raw.log_odds.dtype)))
        if is_dataclass(raw):
            derived=replace(raw,indices=indices,log_odds=odds)
        else:
            derived=copy.copy(raw);derived.indices=indices;derived.log_odds=odds
        self._key,self._snapshot=key,derived
        return derived


class VehicleAwareMissionController(RflyMissionController):
    @classmethod
    def wrap(cls, controller, objects):
        result=object.__new__(cls)
        result.__dict__.update(controller.__dict__)
        result._vehicle_objects=objects
        return result

    def return_and_land(self, *, force_return=False):
        nav=self.semantic_navigator
        if nav is None:
            return super().return_and_land(force_return=force_return)
        original_map,original_vehicle,original_nav_vehicle=nav.global_map,self.vehicle,nav.vehicle
        nav.global_map=VehiclePlanningMap(original_map,self._vehicle_objects,nav)
        self.vehicle=VehicleMotionProxy(original_vehicle,self._vehicle_objects,nav.config)
        nav.vehicle=self.vehicle
        try:
            return super().return_and_land(force_return=force_return)
        finally:
            nav.global_map=original_map
            nav.vehicle=original_nav_vehicle
            self.vehicle=original_vehicle
