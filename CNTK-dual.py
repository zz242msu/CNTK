import cupy as cp
import numpy as np
import argparse
import scipy.linalg
from utils import load_cifar
import threading 

print("CNTK!")
parser = argparse.ArgumentParser(description = 'Convolutional Neural Tangent Kernel (CNTK) for CIFAR-10')
parser.add_argument('--depth', default = 21, type = int, help = 'depth of CNTK (#conv layers + 1)')
parser.add_argument('--gap', default = "yes", type = str, help = 'whether GAP (global average pooling) is used')
parser.add_argument('--fix', default = "yes", type = str, help = 'whether first layer and last layer are fixed (or trained) (see Section 4.2 in our paper)')
args = parser.parse_args()

d = args.depth
gap = (args.gap == "yes")
fix = (args.fix == "yes")

#CUDA kernel for convolution operation
print('conv3')
conv3 = cp.RawKernel(r'''
extern "C" __global__
void conv3(const float s[32][32][32][32], float t[32][32][32][32])
{
	int x1 = threadIdx.x + blockIdx.x - 31;
	int y1 = threadIdx.y + blockIdx.y - 31;
	int x2 = threadIdx.x;
	int y2 = threadIdx.y;

	__shared__ float d[32 + 2][32 + 2];
	if (x2 == 0){
		d[0][y2 + 1] = d[33][y2 + 1] = 0;
		if (x2 == 0 && y2 == 0)
			d[0][0] = d[0][33] = d[33][0] = d[33][33] = 0; 
	}
	if (y2 == 0){
		d[x2 + 1][0] = d[x2 + 1][33] = 0;
	}

	if (x1 < 0 || x1 > 31 || y1 < 0 || y1 > 31){
		d[x2 + 1][y2 + 1] = 0;
		return;
	}
	else
		d[x2 + 1][y2 + 1] = s[x1][y1][x2][y2];
	__syncthreads();

	t[x1][y1][x2][y2] = d[x2][y2] + d[x2][y2 + 1] + d[x2][y2 + 2]
					  + d[x2 + 1][y2] + d[x2 + 1][y2 + 1] + d[x2 + 1][y2 + 2]
					  + d[x2 + 2][y2] + d[x2 + 2][y2 + 1] + d[x2 + 2][y2 + 2];

}''', 'conv3')
conv_blocks = (63, 63)
conv_threads = (32, 32)

#CUDA kernel for activation
print('trans')
trans = cp.RawKernel(r'''
extern "C" __global__
void trans(float s[32][32][32][32], float t[32][32][32][32], const float l[32][32], const float r[32][32], const float il[32][32], const float ir[32][32])
{
	int x1 = blockIdx.x;
	int y1 = blockIdx.y;
	int x2 = threadIdx.x + ((blockIdx.z >> 2) << 3);
	int y2 = threadIdx.y + ((blockIdx.z & 3) << 3);
	float S = s[x1][y1][x2][y2], T = t[x1][y1][x2][y2], L = l[x1][y1], R = r[x2][y2], iL = il[x1][y1], iR = ir[x2][y2];
	S = S * iL * iR;
	float BS = (S * (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) + sqrtf(1.0f - min(S * S, 1.0f))) * L * R / 28.274333882308138f;
	S = (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) / 28.274333882308138;
	t[x1][y1][x2][y2] = T * S + BS;
	s[x1][y1][x2][y2] = BS;

}''', 'trans')
trans_blocks = (32, 32, 16)
trans_threads = (8, 8)

#Calculate diagonal entries of $\Sigma^{(h)}(x, x)$ and their reciprocals. See Section 4.3 in our paper. 
def xx(x):
	RL = [1.0, ]
	iRL = [1.0, ]

	S = cp.matmul(x.T, x).reshape(32, 32, 32, 32)
	conv3(conv_blocks, conv_threads, (S, S))
	T = cp.zeros((32, 32, 32, 32), dtype = cp.float32)
	if not fix:
		T += S

	for i in range(1, d - 1):
		L = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
		iL = 1.0 / L
		RL.append(L)
		iRL.append(iL)
		trans(trans_blocks, trans_threads, (S, T, L, L, iL, iL))		
		conv3(conv_blocks, conv_threads, (S, S))
		conv3(conv_blocks, conv_threads, (T, T))

	L = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
	iL = 1.0 / L
	RL.append(L)
	iRL.append(iL)
	trans(trans_blocks, trans_threads, (S, T, L, L, iL, iL))	
	
	if fix:
		T -= S
	return RL, iRL

#Caclulate the kernel value of x and z.
#Lx and Lz are diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$. 
#iLx and iLz are reciprocals of diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$. 
def xz(x, z, Lx, Lz, iLx, iLz):
	S = cp.matmul(x.T, z).reshape(32, 32, 32, 32)
	conv3(conv_blocks, conv_threads, (S, S))
	T = cp.zeros((32, 32, 32, 32), dtype = cp.float32)
	if not fix:
		T += S

	for i in range(1, d - 1):
		trans(trans_blocks, trans_threads, (S, T, Lx[i], Lz[i], iLx[i], iLz[i]))		
		conv3(conv_blocks, conv_threads, (S, S))
		conv3(conv_blocks, conv_threads, (T, T))

	trans(trans_blocks, trans_threads, (S, T, Lx[-1], Lz[-1], iLx[-1], iLz[-1]))	

	if fix:
		T -= S	
	return cp.mean(T) if gap else cp.trace(T.reshape(1024, 1024))

#Load CIFAR-10.
print('load data')
(X_train, y_train), (X_test, y_test) = load_cifar()
X = np.concatenate((X_train, X_test), axis = 0)
N = X.shape[0]
N_train = X_train.shape[0]
N_test = X_test.shape[0]
X = cp.asarray(X).reshape(-1, 3, 1024)

#Calculate diagonal entries.
L = []
iL = []
print('N', N)
for i in range(N):
	Lx, iLx = xx(X[i])	
	L.append(Lx)
	iL.append(iLx)

#####Calculate kernel values.
# #####Below we provide a naive implementation using for-loops.
# #####Parallelize this part according to your specific computing enviroment to utilize multiple GPUs.

# H = np.zeros((N, N), dtype = np.float32)
# print('N', N)
# for i in range(N):
# 	for j in range(N):
# 		H[i][j] = xz(X[i], X[j], L[i], L[j], iL[i], iL[j])

# print('H', H)
#####

# def compute_on_gpu(X, L, iL, start, end, N, H):
#     for i in range(start, end):
#         for j in range(N):
#             H[i - start, j] = xz(X[i], X[j], L[i], L[j], iL[i], iL[j])

# # Split the data for each GPU
# X_gpu1 = cp.asarray(X[:N//2])
# X_gpu2 = cp.asarray(X[N//2:])
# # L_gpu1 = cp.asarray(L[:N//2])
# # L_gpu2 = cp.asarray(L[N//2:])
# L_gpu1 = L[:N//2]
# L_gpu2 = L[N//2:]
# # iL_gpu1 = cp.asarray(iL[:N//2])
# # iL_gpu2 = cp.asarray(iL[N//2:])
# iL_gpu1 = iL[:N//2]
# iL_gpu2 = iL[N//2:]

# # Allocate memory for partial results on each GPU
# H_gpu1 = cp.zeros((N//2, N), dtype=cp.float32)
# H_gpu2 = cp.zeros((N//2, N), dtype=cp.float32)

# # Use separate threads or processes to run on each GPU
# import threading

# thread1 = threading.Thread(target=compute_on_gpu, args=(X_gpu1, L_gpu1, iL_gpu1, 0, N//2, N, H_gpu1))
# thread2 = threading.Thread(target=compute_on_gpu, args=(X_gpu2, L_gpu2, iL_gpu2, N//2, N, N, H_gpu2))

# thread1.start()
# thread2.start()

# thread1.join()
# thread2.join()

# # Transfer results back to CPU and combine
# H_cpu1 = cp.asnumpy(H_gpu1)
# H_cpu2 = cp.asnumpy(H_gpu2)
# H = np.vstack((H_cpu1, H_cpu2))

# Prepare and Compute Function for each GPU
def prepare_and_compute_on_gpu(gpu_id, X_full, L_full, iL_full, N, H):
    with cp.cuda.Device(gpu_id):
        X_gpu = X_full[gpu_id * N//2 : (gpu_id + 1) * N//2]
        L_gpu = L_full[gpu_id * N//2 : (gpu_id + 1) * N//2]
        iL_gpu = iL_full[gpu_id * N//2 : (gpu_id + 1) * N//2]
        H_gpu = cp.zeros((N//2, N), dtype=cp.float32)

        for i in range(N//2):
            for j in range(N):
                H_gpu[i, j] = xz(X_gpu[i], X_full[j], L_gpu[i], L_full[j], iL_gpu[i], iL_full[j])

        H_cpu = cp.asnumpy(H_gpu)
        H[gpu_id * N//2 : (gpu_id + 1) * N//2, :] = H_cpu

# Initialize H matrix
H = np.zeros((N, N), dtype=np.float32)

# Start Threads for each GPU
print('threading')
threads = []
for gpu_id in range(2):
    thread = threading.Thread(target=prepare_and_compute_on_gpu, args=(gpu_id, X, L, iL, N, H))
    thread.start()
    threads.append(thread)

# Wait for Threads to Complete
for thread in threads:
    thread.join()
	
#Solve kernel regression.
print('solving kr')
Y_train = np.ones((N_train, 10)) * -0.1
for i in range(N_train):
	Y_train[i][y_train[i]] = 0.9
u = H[N_train:, :N_train].dot(scipy.linalg.solve(H[:N_train, :N_train], Y_train))
print ("test accuracy:", 1.0 * np.sum(np.argmax(u, axis = 1) == y_test) / N_test)
