import sys, os
b=sys.argv[1]; O=f"{os.environ['CLAUDE_JOB_DIR']}/tmp/groups/{b}"
os.chdir(f"/media/dataset_access/go2nus_annotated/{b}/slam_output/erasor2")
sys.argv=["ped_groups.py","--out-dir",O]+sys.argv[2:]
src=open(sys.argv.pop(1) if False else os.environ.get("PG","/media/dataset_access/go2nus_annotated/utown_1_2025-03-07_12-41-18_10_corrected_merged/slam_output/erasor2/ped_groups.py")).read().replace("HERE = Path(__file__).resolve().parent","HERE = Path.cwd()")
exec(compile(src,"ped_groups.py","exec"),{"__name__":"__main__"})
