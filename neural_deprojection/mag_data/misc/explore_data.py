import pynbody as nb
from pynbody.plot.sph import volume
from mayavi import mlab
import glob

if __name__ == '__main__':
    my_file = glob.glob('./*/snap*')[0]
    sim = nb.load(my_file)
    volume(sim.dm)
    volume(sim.stars)
    volume(sim.gas)
    mlab.show()