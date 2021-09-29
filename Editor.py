from PIL import Image, ImageDraw
import os, sys 
from numpy import array, zeros, empty, uint8, matmul, sin, pi
from numpy.linalg import inv 
import matplotlib.pyplot as plt 
from statistics import stdev 

name = str(input("Insert image file name in the same directionary:"))
imag = Image.open(os.path.join(sys.path[0], name), "r").convert("RGB") #paths to myphoto.jpg
width, length = imag.size #defines the matrix dims 


state = int(input("Grayscale or RBG output? (Grayscale = 0, RGB =1)"))
if state == 0:
    gray = imag
if state == 1:
    red, green, blue = imag.split()


"""
I begin by asking the user what type of output he wishes to have, and locating their image within the assigned folder.
"""

method = int(input("Shape of masking: (Crescent = 0, Squares = 1, Crescent and squares = 2, text = 3 [only Butterfly and Bee]"))

if method == 0: #Here the result is a small crescent 
    canvas = Image.new("RGB", (width, length), (0,0,0))
    edi = ImageDraw.Draw(canvas)
    edi.ellipse([(100,100), (1000,500)], fill = (255,255,255))
    edi.ellipse([(93,93), (993,493)], fill = (0,0,0))
    mask = array(canvas.convert("L"), dtype = int)/255

def fill(args): #dummy argument, I just wish for the array to be called and not clug the method sections.
    return array([[100,50], 
    [200,100],
    [400,300],
    [150,400],
    [500,100],
    [420,500],
    [200,130],
    [120,300],
    [300,1000],
    [30,900],
    [300,200],
    [60,90],
    [335,600],
    [250,360],
    [220,550],
    [500,700],
    [400,750],
    [470,670],
    [300,500],
    [300,550],
    [300,600],
    [350,770],
    [500,800],
    [350,500],
    [460,460],
    [123,321],
    [100,960],
    [40,1000]])

if method == 1:
    fill = fill("hi")
    mask = zeros([length, width], dtype = float) #here array carrying the square pixels
    s = int(input("Define pixel count of center-edge length for square pixel:"))  #Here I define the length from center to edge of my 2s x 2s square pixel
    print("Squares will have size: ", 2*s-1, "x", 2*s-1)
    n = 0 
    while n<len(fill): #this while loop generetes a mask with 2s-1 x 2s-1 pixels to cover wmask
        i,j = fill[n,0], fill[n,1] 
        for k in range(-s,s):
            for l in range(-s,s):
                mask[i+l,j+k] = 1
        n+=1



if method == 2:
    fill = fill("hi")
    mask = zeros([length, width], dtype = float) #here array carrying the square pixels
    s = int(input("Define pixel count of center-edge length for square pixel:"))  #Here I define the length from center to edge of my 2s x 2s square pixel
    print("Squares will have size: ", 2*s-1, "x", 2*s-1)
    n = 0 
    while n<len(fill):
        i,j = fill[n,0], fill[n,1] 
        for k in range(-s,s):
            for l in range(-s,s):
                mask[i+l,j+k] = 1
        n+=1
    

    canvas = Image.new("RGB", (width, length), (0,0,0))
    edi = ImageDraw.Draw(canvas)
    edi.ellipse([(100,100), (700,500)], fill = (255,255,255))
    edi.ellipse([(100,100), (697,497)], fill = (0,0,0))
    mask = mask + array(canvas.convert("L"), dtype = int)/255

if method == 3:
    if name == "Butterfly.jpg":
        text = Image.open(os.path.join(sys.path[0], "Text2.png"), "r").convert("RGB") 
    if name == "Bee.jpg":
        text = Image.open(os.path.join(sys.path[0], "Text.png"), "r").convert("RGB") 
    
    mask = array(text.convert("L"), dtype = int)/255
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i,j]>0.3:
                mask[i,j]=1
            if not mask[i,j]>0.3:
                mask[i,j]=0




plt.imshow(mask, cmap = "hot")
plt.show()
corruptedImageMask = Image.fromarray(uint8(plt.cm.gray(mask)*255)).convert("L")


"""
Here the user is allowed to apply a mask. This is in allignment to the purpose of the project, to mask a photo and then recast/fix it using PDE:s. 
There are 3 options, and 3rd being a sum of adding the other two.
"""

def masking(imagesel):
    ita = imagesel.convert("L")
    ara = array(ita, dtype = float)/255
    wmask1 = zeros([length, width], dtype = float) #here I store masked elements for edited image
    
    for i in range(width): #appling the mask onto myphoto using array algebra 
        for j in range(length):
            wmask1[j,i] =ara[j,i] + mask[j,i] #the array has 256 colour scaling elements, starting at 0, so I normalise it and add the mask value
            if wmask1[j,i]>=1: #since a value above 1 or equal to 1 implies a pixel I thereby just re-normalise it
                wmask1[j,i] = 1
            else: #obviously, if it doesn't need to re-normalise it passes
                pass 
    return wmask1 

"""
Simply applying the mask onto the provided image. 
"""


mode = int(input("Mode of solving:(Jacobi=0, ORM=1, Explicit Euler=2, Crank-Nicolson Trapezoidal=3, Implicit Euler=4)"))
""""
Here I ask the user to provide which mode of solving they wish to apply.
"""
def fix(dummyarray, mode = mode, colour =False, std = False): #args are: dummyarray = mask + image grayscaled and normalised, mode is method of solving, colour is original colour, std is statistical deviations



    data = dummyarray 
    Deadp = {} #dictionary to store each element corresponding to a dead pixel
    n=0 #index for dictionary
    for i in range(width): #generating the dictionary with the grid point in the image's array
        for j in range(length):
            if mask[j,i] == 1:
                Deadp[n] = (j,i)
                n+=1
            else:
                pass
    """"
I begin by making a map between the position on (x,y) that the dead pixel and the s:th row of the vector u, following that \laplace{u} = D\partial_t u (D which may or may not be
time dependent). I then proceed to use the keys for my dictionary Deadp to provide the mapping from u to the image and as an iterative placeholder for coodinates (i,j).
Not to cause confusion, S will for future reference be called len(Deadp).
    """
    A = zeros([len(Deadp), len(Deadp)], dtype = int) #zeros as a ground for constructing the Matrix A
    B = zeros([len(Deadp),2], dtype = float) #constant vector, and how many "correct" pixels it neighbours
    inde = 0
    for item in Deadp:
        a, b, c, d = (Deadp[item][0]+1,Deadp[item][1]), (Deadp[item][0]-1, Deadp[item][1]), (Deadp[item][0], Deadp[item][1]+1), (Deadp[item][0], Deadp[item][1]-1) #checking each neighbourhood of (x,y)
        A[inde,inde]=-4 #initial values on diagonal
        for itera in [a,b,c,d]: #for point (i,j) we consider neighbours
            if itera in Deadp.values():
                for k in range(len(Deadp)):
                    if itera == Deadp[k]:
                        A[inde,k] += 1 #if a neighbour is a deadpixel, define 1 on the (i,j):th element
                        break 
            if not itera in Deadp.values():
                A[inde,inde]+=1 #if neighbour not a dead pixel, add 1 onto the diagonal
                B[inde,0]-=data[itera[0], itera[1]] #we add the DBC onto 
                B[inde,1]+=1
        inde+=1
    
    """
Defining b (in code B) and A based on the neighbourhood around each point of interest. I make B Sx2 to also give information how many boundary elements each dead pixel has.
    """
    uvec = zeros([len(Deadp), 1], dtype = float) #defining u as zeros 
    for n in range(len(Deadp)):
        if not B[n,1]==0:
            uvec[n,0] = -B[n,0]/B[n,1] #applying boundary condition
    if mode == 0:
        N = 1000 #iteration count 
        for e in range(N): #looping over "time"
            q=0
            for item in Deadp: #method is to add neighbours if the s:th row is NOT a boundary point
                a, b, c, d = data[Deadp[item][0]+1,Deadp[item][1]], data[Deadp[item][0]-1, Deadp[item][1]], data[Deadp[item][0], Deadp[item][1]+1], data[Deadp[item][0], Deadp[item][1]-1]
                if B[q,1] == 0: #if not a boundary, then perform method in lecture notes
                    uvec[q,0] = 1/4*(a+b+c+d)
                q+=1
            n=0
            for item in Deadp: #reapplies the uvec vector onto the data points
                data[Deadp[item][0], Deadp[item][1]] = uvec[n,0]
                n+=1
        if std == False: #If I don't need Statistical Deviations, then return only data
            return data 


    if mode == 1: #see previous mode for further details
        N = 2000 #iteration count
        omega = 0.63 #defining relaxation factor, user may set omega = 1 for GS method
        for e in range(N):
            q=0
            for item in Deadp: #similar as to before, just a different formula
                a, b, c, d = data[Deadp[item][0]+1,Deadp[item][1]], data[Deadp[item][0]-1, Deadp[item][1]], data[Deadp[item][0], Deadp[item][1]+1], data[Deadp[item][0], Deadp[item][1]-1]
                f = data[Deadp[item][0], Deadp[item][1]] #we also consider the s:th row element here too
                if B[q,1] == 0:
                    uvec[q,0] = omega/4*(a+b+c+d) + (1-omega)*f
                q+=1
            n=0
            for item in Deadp:
                data[Deadp[item][0], Deadp[item][1]] = uvec[n,0]
                n+=1
        if std == False:
            return data 
        
    if mode == 2:
        N = 2000 #iteration count
        dt = 1/N #suppose we go from 0->1 in time
        stepsi = 1 #stepsize of pixel (set to be 1)
        for e in range(N+1):
            Diffusion = 450*(50/45- 1/10*sin(5*e*pi/N)) #oscillating diffusion in time

            n=0
            for item in Deadp:
                data[Deadp[item][0], Deadp[item][1]] = uvec[n,0]
                n+=1
            q=0
            for item in Deadp: #similar as to before with adding elements
                a, b, c, d = data[Deadp[item][0]+1,Deadp[item][1]], data[Deadp[item][0]-1, Deadp[item][1]], data[Deadp[item][0], Deadp[item][1]+1], data[Deadp[item][0], Deadp[item][1]-1]
                f = data[Deadp[item][0], Deadp[item][1]]
                if B[q,0] == 0:
                    uvec[q,0] = f + dt*Diffusion/stepsi**2*(a + b + c + d - 4*f) 
                q+=1
        if std == False:
            return data 
    if mode == 3:
        N = 1000 
        dt = 1/N #time step
        diffusion = 400 #diffusion constant 
        stepsi = 1 #pixel step (set to be 1)
        alpha = dt*diffusion/stepsi**2
        identity = zeros([len(Deadp), len(Deadp)], dtype = int)
        for n in range(len(Deadp)):
            identity[n,n] =1
        invert = inv(identity-1/2*alpha*A) #first matrix fom RHS
        matrix2 = identity + 1/2*alpha*A #second matrix from RHS
        matrix3 = matmul(invert, matrix2) #matrix product of first and second matrix, the product acting on uvec
        uvecprim = uvec #I introduce a prime intermediate uvector to carry on new information to uvec 
        for n in range(len(Deadp)):
            identity[n,n]=1  
        

        for e in range(N):
            uvecprim = matmul(matrix3, uvec) #just doing matrix multiplication (note that a vector can be seen as a nx1 matrix)
            for n in range(len(Deadp)):
                if not B[n,1] == 0:
                    uvecprim[n,0] = uvec[n,0]
            uvec = uvecprim 
        n=0
        for item in Deadp:
            data[Deadp[item][0], Deadp[item][1]] = uvec[n,0]
            n+=1
        if std == False:
            return data


    if mode == 4: #similar as before
        N = 1000 
        dt = 1/N
        diffusion = 250
        stepsi = 1
        alpha = dt*diffusion/stepsi**2
        identity = zeros([len(Deadp), len(Deadp)], dtype = float)
        for n in range(len(Deadp)):
            identity[n,n]=1
        inverted = inv(identity - alpha*A)
        uvecprim = uvec   
        for e in range(N):
            uvecprim = matmul(inverted, uvec)
            for n in range(len(Deadp)):
                if not B[n,1] == 0:
                    uvecprim[n,0] = uvec[n,0]
            uvec = uvecprim 
        n=0
        for item in Deadp:
            data[Deadp[item][0], Deadp[item][1]] = uvec[n,0]
            n+=1
        if std == False:
            return data
        
    """
Below I define Statistical Deviations. It uses the same formula provided in the lecture notes 
    """
    if std == True:
        p=0
        ref = list([0 for a in range(len(Deadp))])
        reference = array(colour.convert("L"))/255 #reference here means the actual correct image without masking, used to compare "quality" using chi-squared method.
        while p<len(Deadp):
            i,j = Deadp[p]
            ref[p]=reference[i,j] 
            p+=1
        std = stdev(ref,sum(ref)/len(ref))**2 #global standard deviation in the image for removed pixels
        q = 0
        ChiSq = 0
        while q<len(Deadp):
            i,j = Deadp[q]
            ChiSq+= (data[i,j] - reference[i,j])**2/std/len(Deadp) #chi-squared for "restored" image.
            q+=1
        return data, ChiSq







def make(imagesel):
    """
I take an array with intensity elements and turns it into a grayscale image
    """
    return Image.fromarray(uint8(plt.cm.gray(imagesel)*255)).convert("L")


def merger(*args):
    """
Maerging images, assuming that the end product will be RGB, thereby args will have length of 3, one arg for each channel.
    """
    return Image.merge("RGB", args)

gamma = int(input("want chi squared, iteration count? (No=0, Yes=1)"))
delta = int(input("Save mask,corrupted, fixed image? (No =0, Yes=1)"))
"""
User choosing what they wish to and wish not to give as an output. On the output I just apply functions so there's nothing special about the rest. 
"""
if state == 0:
    if gamma == 0:
        corrupt = make(masking(gray))
        fixed = make(fix(masking(gray)))
        fixed.show()
    if gamma == 1:
        corrupt = make(masking(gray))
        newgray,chigray = fix(masking(gray), colour = gray, std = True)
        fixed = make(newgray)
        print("Chi Squared for grayscale: ", chigray, "With dead pixel count: ")
        fixed.show()

if state == 1:
    if gamma == 0:
        corrupt = merger(make(masking(red)), make(masking(green)), make(masking(blue)))
        fixed = merger(make(fix(masking(red))), make(fix(masking(green))), make(fix(masking(blue))))
        fixed.show()
        
    if gamma == 1:
        corrupt = merger(make(masking(red)), make(masking(green)), make(masking(blue)))
        newred,chired = fix(dummyarray=masking(red), colour = red, std = True)
        newblue,chiblue = fix(dummyarray=masking(blue), colour = blue, std = True)
        newgreen,chigreen = fix(dummyarray = masking(green), colour = green, std = True)
        newred, newblue, newgreen = make(newred), make(newblue), make(newgreen)
        fixed = merger(newred, newgreen, newblue)
        fixed.show()
        print("Chi Squared for red: ", chired)
        print("Chi Squared for green: ", chigreen)
        print("Chi Squared for blue: ", chiblue)


if delta == 1:
    fixed.save("_Restored_" + name)
    corrupt.save("_masked_Image_" + name)
    corruptedImageMask.save("_Mask_" + name)
if delta == 0:
    pass 