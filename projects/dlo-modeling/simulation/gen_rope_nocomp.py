# generate_rope_chain.py
COUNT    = 70         # number of segments
SPACING  = 0.01       # meters per segment
RADIUS   = 0.01       # capsule radius
DAMPING  = 0.005      # per ball joint
FRICTION = "2.0 0.1 0.01"  # slide, twist, roll

hdr = f'''<mujoco model="RopeNoComposite">
  <option timestep="0.002" solver="Newton" iterations="30" tolerance="1e-10"
          jacobian="dense" cone="pyramidal"/>
  <size nconmax="2000" njmax="20000" nstack="500000"/>
  <worldbody>
    <geom name="ground" type="plane" size="0 0 1"/>
    <body name="link_0" pos="0 0 0">
    <freejoint/>
    <geom type="capsule" fromto="0 0 0  0 0 {SPACING}" size="{RADIUS}" friction="{FRICTION}" rgba="0.8 0.2 0.1 1"/>
'''

mid = ""
for i in range(1, COUNT):
    mid += f'''{"  "*(i+3)}<body name="link_{i}" pos="0 0 {SPACING}">
{"  "*(i+4)}<joint name="j_{i}" type="ball" damping="{DAMPING}"/>
{"  "*(i+4)}<geom type="capsule" fromto="0 0 0  0 0 {SPACING}" size="{RADIUS}" friction="{FRICTION}" rgba="0.8 0.2 0.1 1"/>
'''

# close nested bodies
tail_close = ""
for i in range(COUNT, 0, -1):
    tail_close += f'{"  "*(i+2)}</body>\n'

tail = tail_close + '''    </body>
  </worldbody>
</mujoco>
'''

with open("rope_chain.xml", "w") as f:
    f.write(hdr + mid + tail)
print("Wrote rope_chain.xml")