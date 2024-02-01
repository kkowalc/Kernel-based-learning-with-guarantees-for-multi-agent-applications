import matplotlib.pyplot as plt
import matplotlib        as mpl
import numpy as np
from matplotlib import cm
import networkx as nx
from scipy.stats import multivariate_normal
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

plt.close("all")
plt.rcParams.update({'font.size': 22})
np.random.seed(1)

M =25 #number of agents
n=5000
p=2
d=1
h=0.15
delta=0.001
L=0.3
fun=lambda x:(multivariate_normal.pdf(x, mean=[0,0], cov=[0.5,0.5])+multivariate_normal.pdf(x, mean=[1,2], cov=[0.55,0.55])+multivariate_normal.pdf(x, mean=[2,-2], cov=[0.7,0.7]))
Kernel_fun=lambda x :0.5*(x<1).astype(float)

grid=np.array([[-2,-2],
                  [2,2]])
grid_step=np.array([0.25,0.25]) # grid step for every explanatory sub data d x 1
mesh=np.array(np.meshgrid(*[np.arange(l[0], l[1]+grid_step[i],grid_step[i]) for i,l in enumerate(grid.T) ]))
mesh_ind=np.array(np.meshgrid(*[np.arange(0, mesh.shape[i+1],1) for i in range(mesh.shape[0]) ]))
mesh_plot=np.array(np.meshgrid(*[np.arange(l[0], l[1]+0.01,0.01*np.ones_like(grid_step)[i]) for i,l in enumerate(grid.T) ]))


class agent():

    def __init__(self, ID, neighbors,domain,noise_var,mesh,mesh_ind,explanatory_EV,explanatory_var):
        self.ID = ID
        self.neighbors=neighbors
        self.domain=domain
        self.noise_var=noise_var
        self.explanatory_EV=explanatory_EV
        self.explanatory_var=explanatory_var
        self.local_explanatory_data=np.nan * np.empty((p,n))
        self.local_noise_values=np.nan * np.empty((n,d))
        self.local_values=np.nan * np.empty((n,d))
        self.mesh=mesh
        self.mesh_v=self.mesh.T.reshape([self.mesh.size//self.mesh.shape[0],self.mesh.shape[0]])
        self.mesh_ind=mesh_ind
        self.mesh_ind_v=self.mesh_ind.T.reshape([self.mesh_ind.size//self.mesh_ind.shape[0],self.mesh_ind.shape[0]])
        self.local_est=np.zeros(mesh.shape[1:])
        self.acquired_data=np.zeros(shape=[self.mesh.shape[1]*self.mesh.shape[2],M, 2+p+d])
        for m in range(M):
            self.acquired_data[:,m,2+d:2+d+p]=self.mesh_ind_v
        self.acquired_data_tmp=np.zeros(shape=[M, 2+p+d])
        self.local_kappas=np.zeros(mesh.shape[1:])
        self.local_nums=np.zeros(mesh.shape[1:])
        self.final_kappas=np.zeros(mesh.shape[1:])
        self.final_nums=np.zeros(mesh.shape[1:])
        self.final_est=np.zeros(mesh.shape[1:])
        self.final_est_count=np.zeros(mesh.shape[1:])
        self.final_bounds=np.zeros(mesh.shape[1:])
        self.local_bounds=np.zeros(mesh.shape[1:])
        self.h_max=np.zeros(mesh.shape[1:])
        self.H=np.zeros(mesh.shape[1:])
        
        self.obtain_observations()
    def NW(self,data,value,mesh,h,delta,L,noise_var, ker_NW):


        m_tmp=np.array([(mesh[i].reshape(mesh[i].shape+(1,))-data[i].reshape((1,)*mesh[i].ndim+(len(data[i]),)))/h
                        for i in range(mesh.shape[0])]) # stucked arrays of diferences grid - explanatory data
        kernel_values=ker_NW(np.linalg.norm(m_tmp,axis=0))
        num = np.sum(np.multiply(value, kernel_values), axis=mesh.ndim-1)
        kappa = np.sum(kernel_values, axis=(mesh.ndim-1))
        alpha = np.zeros(mesh.shape[1:])
        alpha[kappa <= 1] = np.sqrt(np.log(np.sqrt(2)/delta))
        alpha[kappa > 1] = np.sqrt(kappa[kappa > 1]*np.log(np.sqrt(1 + kappa[kappa > 1])/delta))
        np.seterr(divide='ignore')
        self.local_bounds= L*h + 2*noise_var*alpha/kappa
        return num,kappa

    def obtain_observations(self):
        self.local_explanatory_data=np.random.multivariate_normal(self.explanatory_EV,self.explanatory_var,n).T
        self.local_noise_values=np.random.normal(0,self.noise_var, n)
        self.local_values=fun(self.local_explanatory_data.T)+self.local_noise_values

    def send(self,t):
        if(np.random.uniform(0,1)>0.3) and np.sum(self.acquired_data)>0:
            tuples=np.stack(self.acquired_data)
            ind=np.random.randint([0,0],[tuples.shape[0],tuples.shape[1]])
            #print(ind)
            data_to_send=self.acquired_data[tuple(ind)]
        else:
            if np.sum(self.local_kappas)==0 or (np.random.uniform(0,1)>0.9):
                self.local_nums,self.local_kappas=self.NW(self.local_explanatory_data[:,0:t], self.local_values[0:t], self.mesh, h
                                                             , delta, L, self.noise_var, Kernel_fun)
            indexes=np.vstack(np.where(self.local_kappas>np.max(self.local_kappas)*0.1))
            if indexes.size==0:
                return
            indx=indexes[:,np.random.randint(indexes.shape[1])]
            num_send=self.local_nums[tuple(indx)]
            kappa_send=self.local_kappas[tuple(indx)]
            h_send=1
            #print(num_send)
            data_to_send=[num_send,kappa_send,h_send,*indx]

        for neighbor in self.neighbors:
            agents[neighbor].acquired_data_tmp[self.ID]=data_to_send

    def update(self):
        for i,l in enumerate(self.acquired_data_tmp):
            ind=np.where((self.acquired_data[:,i,2+d:2+d+p]==l[2+d:2+d+p]).all(axis=1))
            if self.acquired_data[ind,i,1]<l[1]:
                self.acquired_data[ind,i,:]=l[:]

    def final_estimate(self):
        for i in range(self.acquired_data.shape[0]):
            ind=self.acquired_data[i,0,2+d:2+d+p].astype(int)
            for j in range(M):
                self.final_nums[tuple(ind)]+=self.acquired_data[i,j,0]
                self.final_kappas[tuple(ind)]+=self.acquired_data[i,j,1]
                np.seterr(divide='ignore', invalid='ignore')
                self.final_est=self.final_nums/self.final_kappas
                lower = (self.final_kappas <= 1) * np.sqrt(np.log(np.sqrt(2) / delta))
                upper = (self.final_kappas > 1) * np.sqrt(self.final_kappas * np.log(np.sqrt(1 + self.final_kappas) / delta))
                alpha = lower + upper
                np.seterr(divide='ignore')
                self.final_bounds=L*h+2*self.noise_var*alpha/self.final_kappas

def random_connection_matrix(M,density):
    NoC=int(M*M*density)
    conn=np.random.randint(0, M, (2,NoC))
    MoC=np.zeros([M,M])
    MoC[conn[0],conn[1]]=1
    MoC[conn[1],conn[0]]=1
    np.fill_diagonal(MoC,0)
    return MoC

def plot_graph(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    all_rows = range(0, adjacency_matrix.shape[0])
    for n in all_rows:
        gr.add_node(n)
    gr.add_edges_from(edges)
    nx.draw(gr,font_size=30, node_size=4000,with_labels=True)
    plt.show()

def random_domain(minv,maxv,M):
    d=len(minv)
    domains=np.random.uniform(minv,maxv,(M,2,d))
    return domains


def NW(data,value,mesh,h,delta,L,noise_var, ker_NW):


    m_tmp=np.array([(mesh[i].reshape(mesh[i].shape+(1,))-data[i].reshape((1,)*mesh[i].ndim+(len(data[i]),)))/h
                    for i in range(mesh.shape[0])]) # stucked arrays of diferences grid - explanatory data
    kernel_values=ker_NW(np.linalg.norm(m_tmp,axis=0))
    num = np.sum(np.multiply(value, kernel_values), axis=mesh.ndim-1)
    kappa = np.sum(kernel_values, axis=(mesh.ndim-1))
    alpha = np.zeros(mesh.shape[1:])
    alpha[kappa <= 1] = np.sqrt(np.log(np.sqrt(2)/delta))
    alpha[kappa > 1] = np.sqrt(kappa[kappa > 1]*np.log(np.sqrt(1 + kappa[kappa > 1])/delta))

    return num,kappa

IDs=np.arange(0,M,1)

MoC=random_connection_matrix(M,0.4)
plot_graph(MoC,IDs)
domains=random_domain([-2.5,-2.5],[2.5,2.5],M)

noise_var=0.05*np.ones(M)
scl=1

agents_EV=np.array(np.meshgrid(np.linspace(scl*-2,scl*2,5)+np.random.normal(0,0.1,5),np.linspace(scl*-2,scl*2,5)+np.random.normal(0,0.1,5))).reshape(2,1,25)



agents = [ agent(ID,np.where(MoC[ID]==1)[0],domains[ID],noise_var[ID],mesh,mesh_ind,agents_EV[:,0,ID],np.diag(np.full(domains[ID].shape[1],np.random.uniform(0.1,0.3)))) for ID in IDs]
indxes=[4,11,13,23]
bounds=np.zeros([4,n])
kappa=0

for t in tqdm(range(n)):
    kappa=0
    for agent in agents:
        agent.update()
        agent.send(t)
    for i,l in enumerate(indxes):
        ind=agents[l].acquired_data[0,0,2+d:2+d+p].astype(int)
        for j in range(M):
            kappa+=agents[l].acquired_data[0,j,1]
            np.seterr(divide='ignore', invalid='ignore')
            lower = (kappa <= 1) * np.sqrt(np.log(np.sqrt(2) / delta))
            upper = (kappa > 1) * np.sqrt(kappa * np.log(np.sqrt(1 + kappa) / delta))
            alpha = lower + upper
            np.seterr(divide='ignore')
            bounds[i,t]=L*h+2*agents[l].noise_var*alpha/kappa
        kappa=0

for agent in agents:
    agent.final_estimate()

global_explanatory_data=np.array([]).reshape(p,0)
global_values=np.array([])
for agent in agents:
    global_explanatory_data=np.hstack((global_explanatory_data,agent.local_explanatory_data))
    global_values=np.hstack((global_values,agent.local_values))

global_nums,global_kappas=NW(global_explanatory_data,global_values,mesh,h,delta,L,noise_var[0],Kernel_fun)
global_est=global_nums/global_kappas
lower = (global_kappas <= 1) * np.sqrt(np.log(np.sqrt(2) / delta))
upper = (global_kappas > 1) * np.sqrt(global_kappas * np.log(np.sqrt(1 + global_kappas) / delta))
alpha = lower + upper
global_bounds=L*h+2*noise_var[0]*alpha/global_kappas
ind=13



ax = plt.figure().add_subplot(projection='3d')
ax.view_init(elev=33., azim=-44)
surf=ax.plot_surface(mesh_plot[0], mesh_plot[1], fun(mesh_plot.T).T, rstride=50, cstride=50,alpha=0.4, color="C0",linewidth=0,edgecolors="C0",lw=0.4,label=r"nonlinearity $m$")
surf2=ax.plot_surface(agents[ind].mesh[0], agents[ind].mesh[1], agents[ind].final_est, color="limegreen",alpha=0.7,antialiased=False,label=r"estimation $\hat{m}$")

ax.scatter(agents[ind].local_explanatory_data[0][::40],agents[ind].local_explanatory_data[1][::40],agents[ind].local_values[::40],color="darkcyan",label=r"agent's local points")

slice_xval=0
slice_yval=0
ind_mp=np.where(mesh_plot[0,:,:]>=slice_xval)[1][0]
ind_m=np.where(agents[ind].mesh[0,:,:]>=slice_xval)[1][0]
zsx=-3
zsy=3


ax.add_collection3d(ax.fill_between(agents[ind].mesh[1][:,ind_m], agents[ind].final_est[ind_m,:]- agents[ind].final_bounds[ind_m,:], agents[ind].final_est[ind_m,:]+ agents[ind].final_bounds[ind_m,:], alpha=0.1,color="blue") ,zs=zsx,zdir="x")
ax.plot(mesh_plot[1][:,ind_mp],fun(mesh_plot.T).T[ind_mp,:],'--',linewidth=1.5,zs=zsx,zdir="x",color="black")
ax.plot(mesh_plot[1][:,ind_mp],fun(mesh_plot.T).T[ind_mp,:],'--',linewidth=2.5,zs=0,zdir="x",color="black",label=r"cross-section")
ax.plot( agents[ind].mesh[1][:,ind_m],agents[ind].final_est[ind_m,:],linewidth=2.5,zs=zsx,zdir="x",color="limegreen")


ind_mp=np.where(mesh_plot[1,:,:]>=slice_yval)[0][0]
ind_m=np.where(agents[ind].mesh[1,:,:]>=slice_yval)[0][0]
ax.add_collection3d(ax.fill_between(agents[ind].mesh[0][ind_m,:],agents[ind].final_est[:,ind_m]- agents[ind].final_bounds[:,ind_m], agents[ind].final_est[:,ind_m]+ agents[ind].final_bounds[:,ind_m], alpha=0.1,color="blue",label='_nolegend_') ,zs=zsy,zdir="y")
ax.plot(mesh_plot[0][ind_mp,:],fun(mesh_plot.T)[ind_mp,:],'--',linewidth=1.5,zs=zsy,zdir="y",color="black")
ax.plot(mesh_plot[0][ind_mp,:],fun(mesh_plot.T)[ind_mp,:],'--',linewidth=2.5,zs=0,zdir="y",color="black",label='_nolegend_')
ax.plot( agents[ind].mesh[0][ind_m,:],agents[ind].final_est[:,ind_m],linewidth=2.5,zs=zsy,zdir="y",color="limegreen")
ax.set_ylim(-2, 3)
ax.set_xlim(-3, 2)
surf3= ax.plot_surface(np.array([0]), np.array([0]), np.array([[0],[0]]), alpha=0.3,color="blue",label=r"error bounds")
surf3._edgecolors2d = surf3._edgecolor3d
surf3._facecolors2d = surf3._facecolor3d
surf2._edgecolors2d = surf2._edgecolor3d
surf2._facecolors2d = surf2._facecolor3d
surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d
ax.legend(loc='upper right')

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.view_init(elev=33., azim=-44)
surf=ax.plot_surface(mesh_plot[0], mesh_plot[1], fun(mesh_plot.T).T, rstride=50, cstride=50,alpha=0.4, color="C0",linewidth=0,edgecolors="C0",lw=0.4,label=r"nonlinearity $m$")
surf2=ax.plot_surface(agents[ind].mesh[0], agents[ind].mesh[1], agents[ind].final_est, color="limegreen",alpha=0.7,antialiased=False,label=r"agents' estimation")
slice_xval=0
slice_yval=0
ind_mp=np.where(mesh_plot[0,:,:]>=slice_xval)[1][0]
ind_m=np.where(agents[ind].mesh[0,:,:]>=slice_xval)[1][0]
zsx=-3
zsy=3


ax.set_zlim3d(0, 0.35)
ax.add_collection3d(ax.fill_between(agents[ind].mesh[1][:,ind_m], agents[ind].final_est[ind_m,:]- agents[ind].final_bounds[ind_m,:], agents[ind].final_est[ind_m,:]+ agents[ind].final_bounds[ind_m,:], alpha=0.1,color="blue") ,zs=zsx,zdir="x")
ax.plot(mesh_plot[1][:,ind_mp],fun(mesh_plot.T).T[ind_mp,:],'--',linewidth=1.5,zs=zsx,zdir="x",color="black")
ax.plot(mesh_plot[1][:,ind_mp],fun(mesh_plot.T).T[ind_mp,:],'--',linewidth=2.5,zs=0,zdir="x",color="black")
ax.plot( agents[ind].mesh[1][:,ind_m],agents[ind].final_est[ind_m,:],linewidth=2.5,zs=zsx,zdir="x",color="limegreen")


ind_mp=np.where(mesh_plot[1,:,:]>=slice_yval)[0][0]
ind_m=np.where(agents[ind].mesh[1,:,:]>=slice_yval)[0][0]
ax.add_collection3d(ax.fill_between(agents[ind].mesh[0][ind_m,:],agents[ind].final_est[:,ind_m]- agents[ind].final_bounds[:,ind_m], agents[ind].final_est[:,ind_m]+ agents[ind].final_bounds[:,ind_m], alpha=0.1,color="blue",label='_nolegend_') ,zs=zsy,zdir="y")
ax.plot(mesh_plot[0][ind_mp,:],fun(mesh_plot.T)[ind_mp,:],'--',linewidth=1.5,zs=zsy,zdir="y",color="black")
ax.plot(mesh_plot[0][ind_mp,:],fun(mesh_plot.T)[ind_mp,:],'--',linewidth=2.5,zs=0,zdir="y",color="black",label='_nolegend_')
ax.plot( agents[ind].mesh[0][ind_m,:],agents[ind].final_est[:,ind_m],linewidth=2.5,zs=zsy,zdir="y",color="limegreen")


handles, labels = ax.get_legend_handles_labels()
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.view_init(elev=33., azim=-44)

surf3=ax.plot_surface(mesh[0], mesh[1], global_est, color="salmon",alpha=0.4,antialiased=False,label=r"global estimation")
ax.plot_surface(mesh_plot[0], mesh_plot[1], fun(mesh_plot.T).T, rstride=50, cstride=50,alpha=0.5, color="C0",linewidth=1,edgecolors="C0",lw=0.4,zorder=100)

slice_xval=0
slice_yval=0
ind_mp=np.where(mesh_plot[0,:,:]>=slice_xval)[1][0]
ind_m=np.where(agents[ind].mesh[0,:,:]>=slice_xval)[1][0]
zsx=-3
zsy=3



ax.add_collection3d(ax.fill_between(agents[ind].mesh[1][:,ind_m], global_est[ind_m,:]- global_bounds[ind_m,:], global_est[ind_m,:]+ global_bounds[ind_m,:], alpha=0.1,color="blue") ,zs=zsx,zdir="x")
ax.plot(mesh_plot[1][:,ind_mp],fun(mesh_plot.T).T[ind_mp,:],'--',linewidth=1.5,zs=zsx,zdir="x",color="black")
ax.plot(mesh_plot[1][:,ind_mp],fun(mesh_plot.T).T[ind_mp,:],'--',linewidth=2.5,zs=0,zdir="x",color="black",label=r"cross-section")
ax.plot( agents[ind].mesh[1][:,ind_m],global_est[ind_m,:],linewidth=2.5,zs=zsx,zdir="x",color="salmon")


ind_mp=np.where(mesh_plot[1,:,:]>=slice_yval)[0][0]
ind_m=np.where(agents[ind].mesh[1,:,:]>=slice_yval)[0][0]
ax.add_collection3d(ax.fill_between(agents[ind].mesh[0][ind_m,:],global_est[:,ind_m]- global_bounds[:,ind_m], global_est[:,ind_m]+global_bounds[:,ind_m], alpha=0.1,color="blue",label='_nolegend_') ,zs=zsy,zdir="y")
ax.plot(mesh_plot[0][ind_mp,:],fun(mesh_plot.T)[ind_mp,:],'--',linewidth=1.5,zs=zsy,zdir="y",color="black")
ax.plot(mesh_plot[0][ind_mp,:],fun(mesh_plot.T)[ind_mp,:],'--',linewidth=2.5,zs=0,zdir="y",color="black",label='_nolegend_')
ax.plot( agents[ind].mesh[0][ind_m,:],global_est[:,ind_m],linewidth=2.5,zs=zsy,zdir="y",color="salmon")
ax.set_zlim3d(0, 0.35)

surf4= ax.plot_surface(np.array([0]), np.array([0]), np.array([[0],[0]]), alpha=0.3,color="blue",label=r"error bounds")
handles2, labels2 = ax.get_legend_handles_labels()

surf4._edgecolors2d = surf4._edgecolor3d
surf4._facecolors2d = surf4._facecolor3d
surf3._edgecolors2d = surf3._edgecolor3d
surf3._facecolors2d = surf3._facecolor3d
surf2._edgecolors2d = surf2._edgecolor3d
surf2._facecolors2d = surf2._facecolor3d
surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d
ax.legend(handles+handles2,labels+labels2,loc="upper center")


bounds[bounds == np.inf] = 1000
plt.figure()
plt.plot(bounds[0,0:-2])
plt.plot(bounds[1,0:-2])
plt.plot(bounds[2,0:-2])
plt.plot(bounds[3,0:-2])
plt.grid()
plt.ylim(0.05,0.2)
plt.xlim(0,5000)
plt.xlabel(r"$t$")
plt.legend([r"$\beta_{4,t}(x)$",r"$\beta_{11,t}(x)$",r"$\beta_{13,t}(x)$",r"$\beta_{23,t}(x)$"])
