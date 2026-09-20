"""Compare the vectorized Genesis BAM equations with upstream BAM."""
from pathlib import Path
import csv, sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from microduck_genesis.bam import bam_motor_torque, bam_voltage

def main():
    # The upstream control and motor equations are scalar/vectorized identical.
    from bam.model import load_model
    model = load_model(motor_name="xl330", model="m6")
    model.actuator.vin = 7.4
    model.actuator.kp = 200.0
    model.actuator.backend = __import__('bam.actuator', fromlist=['NumpyBackend']).NumpyBackend()
    rows=[]; errs=[]
    for q in (-0.5, 0.0, 0.5):
      for qd in (-3.0, 0.0, 3.0):
       for target in (-1.0, 0.0, 1.0):
        vin=7.4
        u=model.actuator.compute_control(target,q,qd,0.02)
        expected=model.actuator.compute_torque(u,True,q,qd)
        t=torch.tensor([target],dtype=torch.float64); qq=torch.tensor([q],dtype=torch.float64); vv=torch.tensor([qd],dtype=torch.float64)
        got=bam_motor_torque(bam_voltage(t,qq,vv,torch.tensor([vin],dtype=torch.float64)),vv).item()
        err=abs(float(expected)-got); errs.append(err)
        rows.append(dict(q=q,qd=qd,target=target,upstream_output=float(expected),genesis_output=got,absolute_error=err,relative_error=err/(abs(float(expected))+1e-9)))
    Path("logs").mkdir(exist_ok=True)
    with open("logs/bam_parity.csv","w",newline="") as f:
      w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"BAM motor max_error={max(errs):.3e} mean_error={sum(errs)/len(errs):.3e}")
    assert max(errs) < 1e-6
if __name__ == "__main__": main()
