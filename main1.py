import subprocess

p1 = subprocess.Popen(["python", "gui.py"])
p2 = subprocess.Popen(["python", "spedsim.py"])
p3 = subprocess.Popen(["python", "speed_comparison.py"])
#p4 = subprocess.Popen(["python","tts.py"])
p1.wait()
p2.wait()
p3.wait()
#p4.wait()