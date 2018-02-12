import numpy as np

step = 0.001 #resolution in 0 calculation, a number closer to 0 means longer calculation time but more accuracy
bound = 5 #t bounds
delta = 0.000000001 #resolution in derivative calculations

def f(x,y):
    return 4*x*y - x**4 - y**4 + 4

def fy(x,y,delta):
    return (f(x,y + delta) - f(x,y))/delta

def fx(x,y, delta):
    return (f(x+delta,y) - f(x,y))/delta

def df(x,y,delta,bound,alpha):
    grad = np.array([fx(x,y,delta),fy(x,y,delta)])
    trange = np.arange(-bound,bound,alpha)
    z = []
    for t in trange:
        z.append(fx(x+grad[0]*t,y+grad[1]*t,delta)*grad[0] - fy(x+grad[0]*t,y+grad[1]*t,delta)*grad[1])

    return z

def nextPoint(guess):
    zs =  abs(np.array(df(guess[0],guess[1],delta,bound,step)))
    least = zs[0]
    index = 0
    for i in range(len(zs)):
        if zs[i] < least:
            least = zs[i]
            index = i
    t = -bound + index*step
    direction = [fx(guess[0],guess[1],delta),fy(guess[0],guess[1],delta)]
    return [guess[0] + direction[0]*t, guess[1] + direction[1]*t]

def estimate(iterations,points):
    if iterations == 0:
        return points
    else:
        points.append(nextPoint(points[-1]))
        return estimate(iterations-1,points)        

