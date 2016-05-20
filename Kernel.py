from bempp.api.integration import gauss_triangle_points_and_weights
from Parameters import *
import shmarray
import multiprocessing
import time


global PROC_COUNT
PROC_COUNT = proc_count()

def Custom_Kernel(sp0, sp1, K, g_0=None, g_1=None):

    if g_0 is None:
        c_0 = np.ones(int(sp0.global_dof_count))
        g_0 = gf(sp0, coefficients=c_0)
    if g_1 is None:
        c_1 = np.ones(int(sp1.global_dof_count))
        g_1 = gf(sp1, coefficients=c_1)



    def evaluate_kernel(global_x, global_y):
        npoints_x = global_x.shape[1]
        npoints_y = global_y.shape[1]
        res = np.zeros([npoints_x, npoints_y], dtype=np.complex128)
        for i in range(npoints_x):
            for j in range(npoints_y):
                x = global_x[:, i]
                y = global_y[:, j]
                res[i, j] = K(x,y)
        return res


    gr0 = sp0.grid
    gr1 = sp1.grid

    el0_0 = list(gr0.leaf_view.entity_iterator(0))
    el2_0 = list(gr0.leaf_view.entity_iterator(2))

    el0_1 = list(gr1.leaf_view.entity_iterator(0))
    el2_1 = list(gr1.leaf_view.entity_iterator(2))

    accuracy_order = bem.global_parameters.quadrature.far.single_order 
    points, weights = gauss_triangle_points_and_weights(accuracy_order)

    N0_0 = len(el0_0)
    N0_1 = len(el0_1)
    N2_0 = len(el2_0)
    N2_1 = len(el2_1)


    t1 = time.time()


    Neval = points.shape[1]
    Phii = np.zeros([9, Neval])
    Phij = np.zeros([9, Neval])
    el0i = el0_0[0]
    k = 0

    for i_dof in range(3):
        for j_dof in range(3):
            dof_values_i = np.zeros(3)
            dof_values_j = np.zeros(3)
            dof_values_i[i_dof] = 1
            dof_values_j[j_dof] = 1
            phii = sp0.evaluate_local_basis(el0i, points, dof_values_i)
            phij = sp1.evaluate_local_basis(el0i, points, dof_values_j)
            Phii[k] = phii
            Phij[k] = phij
            k += 1



    def Loop(i, output_list):
        print i
        el0i = el0_0[i]
        global_dofsi = sp0.get_global_dofs(el0i)

        "Juste pour P1"

        x0 = np.floor(global_dofsi[0]/h).astype(int)
        x1 = np.floor(global_dofsi[1]/h).astype(int)
        x2 = np.floor(global_dofsi[2]/h).astype(int)

        xig = el0i.geometry.local2global(points)
        gii = g_0.evaluate(el0i, points)
        integration_elements_i = el0i.geometry.integration_elements(points)
        Di = Phii * gii * weights * integration_elements_i
        indicesi = np.repeat(global_dofsi, 3)

        for j in range(N0_1):
            el0j = el0_1[j]
            global_dofsj = sp1.get_global_dofs(el0j)
            xjg = el0j.geometry.local2global(points)
            cij = evaluate_kernel(xig, xjg)
            gjj = g_1.evaluate(el0j, points)   
            integration_elements_j = el0j.geometry.integration_elements(points)
            indicesj = np.tile(global_dofsj, 3)
            Dj = Phij * gjj * weights * integration_elements_j
            
            coeff = np.zeros(9, dtype=np.complex128)
            for k in range(9):
                Dij = np.dot(np.array([Di[k, :]]).T, np.array([Dj[k, :]]))
                coeff[k] = np.sum(cij * Dij)
                #output_queue.put((indicesi[k], indicesj[k], np.sum(cij * Dij)))

            output_list[x0].put((indicesi[0:3], indicesj[0:3], coeff[0:3]))
            output_list[x1].put((indicesi[3:6], indicesj[3:6], coeff[3:6]))
            output_list[x2].put((indicesi[6:9], indicesj[6:9], coeff[6:9]))
        

    class Worker(multiprocessing.Process):
      def __init__(self, job_queue, output_list):
        multiprocessing.Process.__init__(self)
        self.job_queue = job_queue
        self.output_list = output_list

      def run(self):
        #New Queue Stuff
        i = None
        while i!='kill':  #this is how I kill processes with queues, there might be a cleaner way.
            i = self.job_queue.get()  #gets a job from the queue if there is one, otherwise blocks.
            if i!='kill':
                Loop(i, self.output_list)
            self.job_queue.task_done()




    class Assign(multiprocessing.Process):
      def __init__(self, C, output_queue):
        self.C = C
        multiprocessing.Process.__init__(self)
        self.output_queue = output_queue

      def run(self):
        job = None
        while job!='kill':
          job = self.output_queue.get()
          if job!='kill':
            ind_i, ind_j, cij = job

            self.C[ind_i, ind_j] += cij
          self.output_queue.task_done()




    C = shmarray.create(shape=(N2_0, N2_1), dtype=np.complex128)

    job_queue = multiprocessing.JoinableQueue()

    proc_W = []
    proc_A = []

     
    Nproc = PROC_COUNT

    h = np.floor(N2_0 / Nproc) + 1 
    output_list = np.empty((Nproc, 0)).tolist()
    for j in range(Nproc):
        output_list[j] = multiprocessing.JoinableQueue()


    for j in range(Nproc):
        proc = Worker(job_queue,output_list) #now we pass the job queue instead
        proc_W.append(proc)
        proc.start()
        
    for j in range(Nproc):
        proc = Assign(C,output_list[j]) #now we pass the job queue instead
        proc_A.append(proc)
        proc.start()


    #all the workers are just waiting for jobs now.
    for i in range(N0_0):
        job_queue.put(i)

    job_queue.join() #this blocks the main process until the queue has been emptied



    #output_queue.join()
    #Now if you want to kill all the processes, just send a 'kill'
    #job for each process.

    for proc_w in proc_W:
        job_queue.put('kill')



    for j in range(Nproc):
        output_list[j].join()
        print j


    job_queue.join()
    print 'waiting'

    for j in range(Nproc):
        output_list[j].put('kill')

    print time.time()-t1, 'total_time'
    return np.array(C)


if __name__ == "__main__":

    def Kernel(x, y):
        #return np.exp(-norm(x-y)**2)
        return 1
        #return np.exp(norm(x))*norm(y)

    def one_function(x, normal, domain_index, result):
        result[0] = 1j



    L = 1
    bem.global_parameters.assembly.boundary_operator_assembly_type = 'dense'

    from Geometry import Square, Sphere
    gr0 = Sphere(L+1)
    gr1 = Sphere(L+1)


    sp0 = fs(gr0)
    sp1 = fs(gr1)

    g_0 = gf(sp0, fun = one_function)
    g_1 = gf(sp1, fun = one_function)



    K = Kernel


    t1 = time.time()
    C = Custom_Kernel(sp0, sp1, K, g_0, g_1)
    from Tensor_Operator import vectensvec
    test = np.dot(np.array([g_0.projections()]).T, np.array([g_1.projections()]))

    print test[0:2, 0:2]
    print C[0:2, 0:2]

    print norm(C-test)