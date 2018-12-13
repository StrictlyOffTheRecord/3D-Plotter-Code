import time
import RPi.GPIO as g
import sys
import cv2
import numpy as np

sys.setrecursionlimit(10000)
g.setwarnings(False)
g.setmode(g.BOARD)
xstep = 33
xdir = 35
ystep = 36
ydir = 37
zstep = 38
zdir = 40
count = 0
delaytime = 0.002
g.setup(xstep, g.OUT)
g.setup(ystep, g.OUT)
g.setup(zstep, g.OUT)
g.setup(xdir, g.OUT)
g.setup(ydir, g.OUT)
g.setup(zdir, g.OUT)
g.output(ydir, g.LOW)
g.output(xdir, g.LOW)
steps = 4
zsteps = 500
cap=cv2.VideoCapture(0)
for i in range(50):
    ret,frame=cap.read()
    if(ret):
        cv2.imshow("vin",frame)
cap.release()
cv2.destroyAllWindows()
frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
frame=frame[:440,40:]
cv2.imshow("g",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
while(1):
    thresh=input("Enter threshold ")
    r,j=cv2.threshold(frame,thresh,255,cv2.THRESH_BINARY_INV)
    cv2.imshow("j",j)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    inp=input("1.Continue  2.Reset")
    if(inp==1):break
the_matrix=j
cv2.imshow("im",the_matrix)
cv2.waitKey(0)
cv2.destroyAllWindows()
rows,columns=np.shape(the_matrix)
print(rows)
print(columns)
count=0
for i  in range(rows):
    for j in range(columns):
        if(the_matrix[i][j]==255):
            the_matrix[i][j]=1
            count+=1
            #print(i," ",j)
print(count)
#rows = 600
#columns = 600
#the_matrix = [[0] * rows for x in range(rows)]
#circle_matrix = [[0 for x in range(columns)] for x in range(rows)]

z_up_waiting_x_coordinate = 0
z_up_waiting_y_coordinate = 0
new_starting_point_x_coordinate = 0
new_starting_point_y_coordinate = 0


x_steps = 0
y_steps = 0

list_size = 0

list=[0]

def x_down(steps):
    g.output(xdir, g.HIGH)
    countx = 0
    while True:
        if (countx > steps):
            ##countx=0
            return False
        g.output(xstep, g.HIGH)
        time.sleep(delaytime)
        g.output(xstep, g.LOW)
        time.sleep(delaytime)
        countx = countx + 1


def x_up(steps):
    g.output(xdir, g.LOW)
    countx = 0
    while True:
        if (countx > steps):
            ##countx=0
            return False
        g.output(xstep, g.HIGH)
        time.sleep(delaytime)
        g.output(xstep, g.LOW)
        time.sleep(delaytime)
        countx = countx + 1


def y_left(steps):
    g.output(ydir, g.LOW)
    county = 0
    while True:
        if (county > steps):
            ##countx=0
            return False
        g.output(ystep, g.HIGH)
        time.sleep(delaytime)
        g.output(ystep, g.LOW)
        time.sleep(delaytime)
        county = county + 1


def y_right(steps):
    g.output(ydir, g.HIGH)
    county = 0
    while True:
        if (county > steps):
            ##countx=0
            return False
        g.output(ystep, g.HIGH)
        time.sleep(delaytime)
        g.output(ystep, g.LOW)
        time.sleep(delaytime)
        county = county + 1


def z_up(zsteps):
    g.output(zdir, g.HIGH)
    countz = 0
    while True:
        if (countz > zsteps):
            ##countx=0
            return False
        g.output(zstep, g.HIGH)
        time.sleep(0.002)
        g.output(zstep, g.LOW)
        time.sleep(0.002)
        countz = countz + 1


def z_down(zsteps):
    g.output(zdir, g.LOW)
    countz = 0
    while True:
        if (countz > zsteps):
            ##countx=0
            return False
        g.output(zstep, g.HIGH)
        time.sleep(0.002)
        g.output(zstep, g.LOW)
        time.sleep(0.002)
        countz = countz + 1


def line_down_left(steps):  ##quad1
    g.output(xdir, g.HIGH)
    g.output(ydir, g.LOW)
    stepsupdate = steps / 1
    countxy = 0
    while True:
        if (countxy > stepsupdate):
            ##countx=0
            return False
        g.output(xstep, g.HIGH)
        g.output(ystep, g.HIGH)
        time.sleep(delaytime)
        g.output(xstep, g.LOW)
        g.output(ystep, g.LOW)
        time.sleep(delaytime)
        countxy = countxy + 1


def line_down_right(steps):  ##quad2
    g.output(xdir, g.HIGH)
    g.output(ydir, g.HIGH)
    stepsupdate = steps / 1
    countxy = 0
    while True:
        if (countxy > stepsupdate):
            ##countx=0
            return False
        g.output(xstep, g.HIGH)
        g.output(ystep, g.HIGH)
        time.sleep(delaytime)
        g.output(xstep, g.LOW)
        g.output(ystep, g.LOW)
        time.sleep(delaytime)
        countxy = countxy + 1


def line_up_right(steps):  ##quad 3
    g.output(xdir, g.LOW)
    g.output(ydir, g.HIGH)
    stepsupdate = steps / 1
    countxy = 0
    while True:
        if (countxy > stepsupdate):
            ##countx=0
            return False
        g.output(xstep, g.HIGH)
        g.output(ystep, g.HIGH)
        time.sleep(delaytime)
        g.output(xstep, g.LOW)
        g.output(ystep, g.LOW)
        time.sleep(delaytime)
        countxy = countxy + 1


def line_up_left(steps):  # quad 4
    g.output(xdir, g.LOW)
    g.output(ydir, g.LOW)
    stepsupdate = steps / 1
    countxy = 0
    while True:
        if (countxy > stepsupdate):
            ##countx=0
            return False
        g.output(xstep, g.HIGH)
        g.output(ystep, g.HIGH)
        time.sleep(delaytime)
        g.output(xstep, g.LOW)
        g.output(ystep, g.LOW)
        time.sleep(delaytime)
        countxy = countxy + 1


""" CREATING A LINKED LIST FOR SMOOTH MOVEMENT OF STEPPER MOTORS """

""""
class node:
    def __init__(self, data = None, next = None):
        self.data = data
        self.next = next


class linked_list:
    def __init__(self):
        self.size = 0
        self.first = None
        self.last = None

    def add_new_node(self, data):
        new_node = node(data)
        if self.size == 0:
            self.first = new_node
            self.first.next = new_node
            self.last = new_node
        else:
            self.last.next = new_node
            self.last = new_node
        self.size += 1

    def print_linked_list(self):
        node = self.first
        while node:
            print(node.data)

    def traverse_linked_list(self):
        node = self.first
        while node:
            smooth_motion(node.data)
            node = node.next
"""
""" LINKED LIST FINISHED """
"""
l1 = linked_list()
"""

def intarray(binstring):
    '''Change a 2D matrix of 01 chars into a list of lists of ints'''
    return [[1 if ch == '1' else 0 for ch in line]
            for line in binstring.strip().split()]


def chararray(intmatrix):
    '''Change a 2d list of lists of 1/0 ints into lines of 1/0 chars'''
    return '\n'.join(''.join(str(p) for p in row) for row in intmatrix)


def toTxt(intmatrix):
    '''Change a 2d list of lists of 1/0 ints into lines of '#' and '.' chars'''
    return '\n'.join(''.join(('#' if p else '.') for p in row) for row in intmatrix)


def neighbours(x, y, the_matrix):
    '''Return 8-neighbours of point p1 of picture, in order'''
    i = the_matrix
    x1, y1, x_1, y_1 = x + 1, y - 1, x - 1, y + 1
    # print ((x,y))
    return [i[y1][x], i[y1][x1], i[y][x1], i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))


def zhangSuen(the_matrix):
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        # Step 1
        changing1 = []
        for y in range(1, len(the_matrix) - 1):
            for x in range(1, len(the_matrix[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, the_matrix)
                if (the_matrix[y][x] == 1 and  # (Condition 0)
                                    P4 * P6 * P8 == 0 and  # Condition 4
                                    P2 * P4 * P6 == 0 and  # Condition 3
                            transitions(n) == 1 and  # Condition 2
                                2 <= sum(n) <= 6):  # Condition 1
                    changing1.append((x, y))
        for x, y in changing1: the_matrix[y][x] = 0
        # Step 2
        changing2 = []
        for y in range(1, len(the_matrix) - 1):
            for x in range(1, len(the_matrix[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, the_matrix)
                if (the_matrix[y][x] == 1 and  # (Condition 0)
                                    P2 * P6 * P8 == 0 and  # Condition 4
                                    P2 * P4 * P8 == 0 and  # Condition 3
                            transitions(n) == 1 and  # Condition 2
                                2 <= sum(n) <= 6):  # Condition 1
                    changing2.append((x, y))
        for x, y in changing2: the_matrix[y][x] = 0
        # print changing1
        # print changing2
    return the_matrix







def print_list():
    i=1
    while(i<=list_size):
        print(list[i])
        i += 1

def traverse_list():
    i = 1
    while (i <= list_size):
        smooth_motion(list[i])
        i += 1


def smooth_motion(function_number):
    if function_number == 1:
        x_down(steps)

    if function_number == 2:
        x_up(steps)

    if function_number == 3:
        y_left(steps)

    if function_number == 4:
        y_right(steps)

    if function_number == 5:
        z_up(zsteps)

    if function_number == 6:
        z_down(zsteps)

    if function_number == 7:
        line_down_left(steps)

    if function_number == 8:
        line_down_right(steps)

    if function_number == 9:
        line_up_right(steps)

    if function_number == 10:
        line_up_left(steps)
		
    if function_number == 11:
		x_down(x_steps)
	
    if function_number == 12:
        x_up(x_steps)
	
    if function_number == 13:
        y_right(y_steps)
	
    if function_number == 14:
        y_left(y_steps)
		   
    if function_number == 15:
        x_up(z_up_waiting_x_coordinate*steps)
        
    if function_number == 16:
        y_left(z_up_waiting_y_coordinate*steps)


def display_the_matrix():
    for i in range(rows):
        for j in range(columns):
            #			print(the_matrix[i][j]) # end to put in same line
            m = 0


# print("\n")


def reset_pen():
    global list_size
    global z_up_waiting_x_coordinate, z_up_waiting_y_coordinate
    global new_starting_point_x_coordinate, new_starting_point_y_coordinate    
    """
	#l1.add_new_node(15)
    i=0
    while (i < z_up_waiting_y_coordinate):
        list.append(2)
        list_size += 1
        i += 1
    #l1.add_new_node(16)
    i=0
    while (i < z_up_waiting_x_coordinate):
        list.append(3)
        list_size += 1
        i += 1
    """
    new_starting_point_x_coordinate=0
    new_starting_point_y_coordinate=0
    reposition_pen()
	
def reposition_pen():
    #print("PEN REPOSITIONING\n")
    #print(z_up_waiting_x_coordinate, ",", z_up_waiting_y_coordinate)
    #print(new_starting_point_x_coordinate, ",", new_starting_point_y_coordinate)

    global x_steps
    global y_steps
    global list_size
    x_steps = 0
    y_steps = 0
    i = 0
    if z_up_waiting_x_coordinate < new_starting_point_x_coordinate:
        y_steps = (new_starting_point_x_coordinate - z_up_waiting_x_coordinate)
        # x_down(x_steps)
        #print("x_down")
        while(i < y_steps):
            list.append(1)
            list_size += 1
            i += 1
        #print(new_starting_point_x_coordinate, ",", z_up_waiting_x_coordinate, " Steps are:", x_steps)
    i = 0
    if z_up_waiting_x_coordinate > new_starting_point_x_coordinate:
        y_steps = (z_up_waiting_x_coordinate - new_starting_point_x_coordinate)
        # x_up(x_steps)
        #print("x_up")
        while (i < y_steps):
            list.append(2)
            list_size += 1
            i += 1
        #print(new_starting_point_x_coordinate, ",", z_up_waiting_x_coordinate, " Steps are:", x_steps)
    i = 0
    if z_up_waiting_y_coordinate < new_starting_point_y_coordinate:
        x_steps = (new_starting_point_y_coordinate - z_up_waiting_y_coordinate)
        # y_right(y_steps)
        #print("y_right")
        while (i < x_steps):
            list.append(4)
            list_size += 1
            i += 1
        #print(new_starting_point_y_coordinate, ",", z_up_waiting_y_coordinate, " Steps are:", y_steps)

    i = 0
    if z_up_waiting_y_coordinate > new_starting_point_y_coordinate:
        x_steps = (z_up_waiting_y_coordinate - new_starting_point_y_coordinate)
        # y_left(y_steps)
        #print("y_left")
        while (i < x_steps):
            list.append(3)
            list_size += 1
            i += 1
        #print(new_starting_point_y_coordinate, ",", z_up_waiting_y_coordinate, " Steps are:", y_steps)





def mayank(v):
    while True:
        ret=traverse_matrix_using_neighbours(v[0],v[1])
        v=ret
        #print(v)
        if ret[0] == -1:
            break

def traverse_matrix_using_neighbours(i, j):

    global z_up_waiting_x_coordinate
    global z_up_waiting_y_coordinate
    global list_size
    v=[0,0]

    # display_the_matrix()
    # print("The point is", i, ",", j)

    if i != rows-1 and (the_matrix[i + 1][j] == 1 or the_matrix[i + 1][j] == 3):
        # x_down(steps)
        #l1.add_new_node(1)
        list.append(1)
        list_size += 1
        # print("x_down")
        the_matrix[i + 1][j] = 2
        #display_the_matrix()
        #traverse_matrix_using_neighbours(i + 1, j)
        v[0]=i+1
        v[1]=j
        return v
    elif j != columns and (the_matrix[i][j + 1] == 1 or the_matrix[i][j + 1] == 3):
        # y_right(steps)
        #l1.add_new_node(4)
        list.append(4)
        list_size += 1
        # print("y_right")
        the_matrix[i][j + 1] = 2
        #display_the_matrix()
        #traverse_matrix_using_neighbours(i, j + 1)
        v[0]=i
        v[1]=j + 1
        return v

    elif i != rows-1 and j != columns-1 and (the_matrix[i + 1][j + 1] == 1 or the_matrix[i + 1][j + 1] == 3):
        # line_down_right(steps)
        #l1.add_new_node(8)
        list.append(8)
        list_size += 1
        # print("line_down_right")
        the_matrix[i + 1][j + 1] = 2
        #display_the_matrix()
        #traverse_matrix_using_neighbours(i + 1, j + 1)
        v[0]=i+1
        v[1]=j + 1
        return v


    elif j != 1 and (the_matrix[i][j - 1] == 1 or the_matrix[i][j - 1] == 3):
        # y_left(steps)
        #l1.add_new_node(3)
        list.append(3)
        list_size += 1
        # print("y_left")
        the_matrix[i][j - 1] = 2
        #display_the_matrix()
#        traverse_matrix_using_neighbours(i, j - 1)
        v[0]=i
        v[1]=j - 1
        return v


    elif i != rows-1 and j != 1 and (the_matrix[i + 1][j - 1] == 1 or the_matrix[i + 1][j - 1] == 3):
        # line_down_left(steps)
        #l1.add_new_node(7)
        list.append(7)
        list_size += 1
        # print("line_down_left")
        the_matrix[i + 1][j - 1] = 2
        #display_the_matrix()
#        traverse_matrix_using_neighbours(i + 1, j - 1)
        v[0]=i+1
        v[1]=j-1
        return v


    elif i != 1 and (the_matrix[i - 1][j] == 1 or the_matrix[i - 1][j] == 3):
        # x_up(steps)
        #l1.add_new_node(2)
        list.append(2)
        list_size += 1
        # print("x_up")
        the_matrix[i - 1][j] = 2
        #display_the_matrix()
#        traverse_matrix_using_neighbours(i - 1, j)
        v[0]=i-1
        v[1]=j
        return v


    elif i != 1 and j != 1 and (the_matrix[i - 1][j - 1] == 1 or the_matrix[i - 1][j - 1] == 3):
        # line_up_left(steps)
        #l1.add_new_node(10)
        list.append(10)
        list_size += 1
        # print("line_up_left")
        the_matrix[i - 1][j - 1] = 2
        #display_the_matrix()
#        traverse_matrix_using_neighbours(i - 1, j - 1)
        v[0]=i-1
        v[1]=j-1
        return v


    elif i != 1 and j != columns-1 and (the_matrix[i - 1][j + 1] == 1 or the_matrix[i - 1][j + 1] == 3):
        # line_up_right(steps)
        #l1.add_new_node(9)
        list.append(9)
        list_size += 1
        # print("line_up_right")
        the_matrix[i - 1][j + 1] = 2
        #display_the_matrix()
#        traverse_matrix_using_neighbours(i - 1, j + 1)
        v[0]=i-1
        v[1]=j+1
        return v

    else:
        # z_up()
        #l1.add_new_node(5)
        the_matrix[i][j] = 3
        list.append(5)
        list_size += 1
        # print("z_up")
        # print("no neighbor found")
        z_up_waiting_x_coordinate = i
        z_up_waiting_y_coordinate = j
        # print(z_up_waiting_x_coordinate, z_up_waiting_y_coordinate) # these values are not saved, find a way to pass them into find_starting_point_in_matrix()
        # print("Searching again")
        v[0]=-1
        #return v
        search_both_ways_in_the_matrix()
        return v


def search_both_ways_in_the_matrix():

    global new_starting_point_x_coordinate
    global new_starting_point_y_coordinate
    global list_size
    global rows
    global columns
    i = rows-1
    j = columns-1
    flag = 0
    v=[0,0]
    for i in range(rows):
        for j in range(columns - 1, -1, -1):
            if the_matrix[i][j] == 1:
                new_starting_point_x_coordinate = i
                new_starting_point_y_coordinate = j
                #print("Found inner below")
                reposition_pen()
                # z_down()
                #l1.add_new_node(6)
                list.append(6)
                list_size += 1
                the_matrix[i][j] = 3
                display_the_matrix()
                #traverse_matrix_using_neighbours(i, j)
                v[0]=i
                v[1]=j
                mayank(v)
                flag = 1

            if flag == 1:
                break
            j -= 1
        if flag == 1:
            break
        i -= 1
"""
    i = x
    j = 0
    while i < rows:
        while j < columns:
            if the_matrix[i][j] == 1:
                new_starting_point_x_coordinate = i
                new_starting_point_y_coordinate = j
                print("Found inner above")
                reposition_pen()
                # z_down()
                l1.add_new_node(6)
                the_matrix[i][j] = 3
                display_the_matrix()
                traverse_matrix_using_neighbours(i, j)
                flag = 1

            if flag == 1:
                break
            j += 1
        if flag == 1:
            break
        i += 1 """



def find_starting_point_in_matrix():
    flag = 0
    v=[0,0]
    global list_size
    global new_starting_point_x_coordinate
    global new_starting_point_y_coordinate
	
    for i in range(rows):
        for j in range(columns - 1, -1, -1):
            #print(i," ",j)
            if the_matrix[i][j] == 1:
                new_starting_point_x_coordinate = i
                new_starting_point_y_coordinate = j
		reposition_pen()
                #print("reposiitoned")
                # z_down(zsteps)
                #l1.add_new_node(6)
                list.append(6)
                list_size += 1
                the_matrix[i][j] = 3
                display_the_matrix()
#                traverse_matrix_using_neighbours(i, j)
                v[0]=i
                v[1]=j
                mayank(v)
                flag = 1
                #print("something done")
            if flag == 1:
                break

        if flag == 1:
            break


def create_circle_matrix(x,y,r):

    make_a_circle_in_the_matrix(x, y, r)

    for i in range(rows):
        for j in range(columns):
            if circle_matrix[i][j] == 1:
                the_matrix[i][j] = 1
			
			
def make_a_circle_in_the_matrix(x, y, r):
    i = 0
    j = 0
    #    print(x, " ", y, " ", r)
    for i in range(0, rows):
        for j in range(0, rows):
            temp = (i - x) * (i - x)
            temp1 = (j - y) * (j - y)
            temp2 = (r * r)
            ##			print(i," ",j," ",temp," ",temp1," ",temp2)
            ##			if(((i-x)*(i-x)+(j-y)*(j-y))<(r*r)):
            if ((temp + temp1) < temp2):
                circle_matrix[i][j] = 1
                ##				print("a")

    for i in range(0, rows):
        for j in range(0, rows):

            temp = (i - x) * (i - x)
            temp1 = (j - y) * (j - y)
            temp2 = ((r - 1) * (r - 1))

            ##        		if(((i-x)*(i-x)+(j-y)*(j-y))<((r-1)*(r-1))):
            if ((temp + temp1) < temp2):
                circle_matrix[i][j] = 0

            ##				print("b")
            j = j + 1
        i = i + 1


def write_list_to_file():
    file = open("Minor_Project.txt", "w")
    i = 0
    while (i <= list_size):
        file.write('%d'%list[i])
        file.write("\n")
        i += 1
    f.close()

def read_list_from_file():
    global test_file_list
    file = open("Minor_Project.txt")
    for line in file:
        list.append([int(i) for i in line.split()])
		
		
try:
    #x_down(100)
    #	x_down(1000)
    # x_down(1000)
    # y_right(1000)
    # z_down(zsteps)
    # z_down(zsteps)
    # z_down(zsteps)
    # z_down(zsteps)
    # z_down(zsteps)
    # z_up(zsteps)
    # line_down_right(1000)
    # line_up_right(1000)

    # line_up_left(1000)
    #x_down(100)
    #x_down(100)
    #x_down(100)
    #x_down(100)
# line_down_left(1000)
#    y_right(30)
#    x_down(500)
#    x_up(3000)
    #z_down(zsteps)
    #z_up(zsteps)
    #create_circle_matrix(300, 100, 100)
    #create_circle_matrix(300, 150, 100)
    #create_circle_matrix(300,200,100)
#    create_circle_matrix(100,75,60)
#    create_circle_matrix(250,75,40)
#   create_circle_matrix(175,200,40)
#    display_the_matrix()

    zhangSuen(the_matrix)
    count=0
    for i  in range(rows):
        for j in range(columns):
            if(the_matrix[i][j]==1):
                count+=1
                #print(i," ",j)
    print(count)
    find_starting_point_in_matrix()
    print("starting position found")
    reset_pen()
    print("starting position found")
    
    enter_key=input("Enter Any Key")
	
#	write_list_to_file()
#	read_list_from_file()

    traverse_list()
    #l1.traverse_linked_list()
# x_up(800)
# x_down(800)
# y_left(1000)
# y_right(1000)



except KeyboardInterrupt:
    g.cleanup()
    exit()



