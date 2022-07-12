# add test_things.py module to path
import sys
import numpy as np
from SimRunner import BitFlipRunner,FredkinRunner
import math

# always include these lines of MPI code!
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()  # number of MPI procs
rank = comm.Get_rank()  # i.d. for local proc


#defaults = {'localization':18., 'location':.5, 'depth':3, 'tilt':2., 'beta':1., 'tau':1., 'scale':1., 'dt':1/10000, #'lambda':1, 'N':10_000}

initial_params = {'N':50_000,'dt':1/5_000,'target_work':None, 'k':1, 'depth':8, 'location':2}


def save_func(self):
   lmda = int(1000*self.params['lambda'])
   N = int(self.params['N'])
   k = int(self.params['k'])
   return [f'k{k}lmda{lmda:03d}N{N}', None]

simrun = BitFlipRunner()
simrun.save_name = save_func
simrun.change_params(initial_params)

#run_name = 'heatmap_sym_detail_2X20L_5X8g_p16p31_N40/'
#save_dir = '/home/kylejray/FQ_sims/results/{}/'.format(run_name)



#max_L = 8
#L_lists = [ L_range[i*max_L:(i+1)*max_L] for i in range(math.ceil(len(L_range)/max_L))]



# my_param = params[rank]

#dt = [1/item for item in [500, 1_000, 5_000, 10_000, 15_000, 20_000, 30_000]][rank]
#n = [item*1000 for item in [1,4,16,64,256,512]][rank]

'''
locals = [ 20, 50, 100, 200]
zs = [ 2, 3, 4, 5, 6]
combos=[]
for l in locals:
   for z in zs:
      loc = z/np.sqrt(2*l)
      combos.append({'localization':l, 'location':loc})

print(len(combos))
c_lists = [ combos[i*size:(i+1)*size] for i in range(math.ceil(len(combos)/size))]

for combos in c_lists:
   params = combos[rank]
'''

p_list=[]
p_keys = ['k', 'lambda']
lmdas = np.linspace(0, .2, 8)
ks = [1,4]
for k in ks:
   for l in lmdas:
      p_list.append({key:val for key,val in zip(p_keys,[k,l])})

p_lists = [ p_list[i*size:(i+1)*size] for i in range(math.ceil(len(p_list)/size))]

for params in p_lists:
   simrun.change_params(params[rank])

   # save parameters to output file by printing
   sys.stdout.write('my rank:{} of {}, params={} '.format(rank+1, size, params))
   # perform local computation using local param
   simrun.run_sim()
   # save your results
   simrun.save_sim()