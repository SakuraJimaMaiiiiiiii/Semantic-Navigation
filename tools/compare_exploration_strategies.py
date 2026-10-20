"""Compare completed exploration reports, not an estimate of full-map coverage."""
import argparse
import json
import math
from pathlib import Path


def summarize(path):
    path=Path(path)
    if path.is_dir():
        path=path/'exploration_report.json' if (path/'exploration_report.json').exists() else path/'exploration'/'exploration_report.json'
    report=json.loads(path.read_text(encoding='utf-8'))
    decisions=report.get('decision_trace',[])
    selections=report.get("selection_trace",[])
    arrived=report.get("arrived_frontiers")
    bearings=[s["travel_heading"] for s in selections]
    visits=[identity for event,identity in report.get('visit_trace',[]) if event=='visit']
    # Cluster revisits are measured by the decision-layer spatial visit history,
    # rather than place IDs (place IDs always increase).
    return dict(run=str(path),strategy=report.get('strategy'),
                backend=report.get('exploration_planner'),reason=report.get('reason'),
                elapsed_seconds=report.get('elapsed_seconds'),
                observed_area_m2=report.get('observed_area_m2'),
                travel_distance_m=report.get('travel_distance_m'),
                arrival_count=len(visits),blocked_frontiers=report.get('blocked_frontiers'),
                replan_count=len(decisions) if decisions else None,
                repeated_arrivals=None if arrived is None else len(arrived)-len(set(arrived)),
                heading_reversals=sum(abs((b-a+math.pi)%(2*math.pi)-math.pi)>=math.radians(150)
                                      for a,b in zip(bearings,bearings[1:])) if selections else None,
                target_switches=sum(bool(d.get('switched')) for d in decisions) if decisions else None,
                revisited_selections=sum(d.get('visit_count',0)>0 for d in decisions) if decisions else None,
                mean_decision_seconds=(sum(d.get('decision_seconds',0) for d in decisions)/len(decisions) if decisions else None),
                returned_home=report.get('returned_home'),
                full_map_coverage=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    text=json.dumps([summarize(path) for path in args.runs],ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:args.output.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
