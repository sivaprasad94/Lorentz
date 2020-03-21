import concurrent.futures
import time

def func(n,t):
	return n+t
def start_thread_pool():
	pool = concurrent.futures.ThreadPoolExecutor(2)		#创建一个总线程数为2的线程池
	li = [0, 1, 2, 3, 4]								#创建待执行线程(5个)
	for n in li:										#在循环中每次取出一个线程来执行func函数，线程池没有空闲线程时则会进入等待。
		y=pool.submit(func, n,2)
        print(y)


if __name__ == '__main__':
	start_thread_pool()
	print("已全部完成")
