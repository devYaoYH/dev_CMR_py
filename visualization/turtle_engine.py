from __future__ import print_function
import sys
import math
import time
import turtle
import random

##########################
##########################
## pyTurtle Drawing Lib ##
##~~~~~ K O D L E ~~~~~~##
##########################
##########################

# Compatibility
if (sys.version_info.major > 2):
    xrange = range
    raw_input = input

##################
# TIMER FUNCTION #
##################
def debug_time(msg, init, now):
    print("{} {}ms".format(msg, int(round((now-init)*1000*1000))/1000.0), file=sys.stderr)

################
# MATH HELPERS #
################
EPSILON = 0.00000001

def absRad(r):
    if (r >= -math.pi and r <= math.pi):
        return r
    return r-(r//math.pi)*math.pi

def vectorFromRad(r):
    return Vector(math.cos(r), math.sin(r))

##############
# PRIMITIVES #
##############
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "({} {})".format(self.x, self.y)

    def __add__(self, o):
        if (isinstance(o, Vector)):
            return Point(self.x+o.vx, self.y+o.vy)
        print("TYPE ERROR (Point-Vector Addition): ", o, file=sys.stderr)
        return None

    def __sub__(self, o):
        if (isinstance(o, Point)):
            return Vector(self.x-o.x, self.y-o.y)
        elif (isinstance(o, Vector)):
            return Point(self.x-o.vx, self.y-o.vy)
        return None

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __abs__(self):
        return math.sqrt(self.x**2+self.y**2)

    def round(self):
        return Point(round(self.x), round(self.y))

    def distTo(self, o):
        if (isinstance(o, Point)):
            return abs(o-self)
        print("TYPE ERROR (distance to point): ", o, file=sys.stderr)
        return None

    def vecTo(self, o):
        if (isinstance(o, Point)):
            return o-self
        print("TYPE ERROR (vector between points): ", o, file=sys.stderr)
        return None

    def dirTo(self, o):
        if (isinstance(o, Point)):
            return self.vecTo(o).rad
        return None

    def setPoint(self, x, y):
        self.x = x
        self.y = y

    def movePt(self, p):
        self.x = p.x
        self.y = p.y

class Vector(object):
    def __init__(self, vx, vy):
        self.v = (vx, vy)
        self.vx = vx
        self.vy = vy
        self.rad = math.atan2(float(vy), float(vx))
        self.length = abs(self)
        self.norm = self
        if (abs(self.length - 1) > EPSILON and self.length != 0):
            self.norm = Vector(self.vx/self.length, self.vy/self.length)

    def __str__(self):
        return "({} {})".format(self.vx, self.vy)

    def __add__(self, o):
        if (isinstance(o, Point)):
            return Point(o.x+self.vx, o.y+self.vy)
        elif(isinstance(o, Vector)):
            return Vector(self.vx+o.vx, self.vy+o.vy)
        print("TYPE ERROR (Vector addition): ", o, file=sys.stderr)
        return None

    def __sub__(self, o):
        if (isinstance(o, Vector)):
            return Vector(self.vx-o.vx, self.vy-o.vy)
        print("TYPE ERROR (Vector subtraction): ", o, file=sys.stderr)
        return None

    def __mul__(self, k):
        if (isinstance(k, int) or isinstance(k, float)):
            return Vector(self.vx*k, self.vy*k)
        print("TYPE ERROR (Vector multiplication with constant): ", k, file=sys.stderr)
        return None

    def __xor__(self, o): # DOT PDT
        if (isinstance(o, Vector)):
            return self.vx*o.vx+self.vy*o.vy
        print("TYPE ERROR (Vector dot-pdt): ", o, file=sys.stderr)
        return None

    def __neg__(self):
        return Vector(-self.vx, -self.vy)

    def __abs__(self):
        return math.sqrt(self.vx**2+self.vy**2)

    def round(self):
        return Vector(round(self.vx), round(self.vy))

    def dirTo(self, o):
        if (isinstance(o, Vector)):
            return absRad(self.rad-o.rad)
        print("TYPE ERROR (Vector angular difference): ", o, file=sys.stderr)
        return None

    def proj(self, o):
        if (isinstance(o, Vector)):
            return o.norm*(self^o.norm)
        print("TYPE ERROR (Vector projection): ", o, file=sys.stderr)
        return None

class Turtle(object):
    def __init__(self, size_x=400, size_y=300, scaling=0.5):
        ##########
        # CONFIG #
        ##########
        # COLORS
        self.col_black = (0, 0, 0)
        self.col_grey = (216, 216, 216)
        self.col_red = (196, 32, 32)
        self.col_green = (32, 196, 32)
        self.col_blue = (32, 32, 196)
        self.col_purple = (196, 32, 196)

        # Setup Display Configuration
        self.SIZE_X = size_x
        self.SIZE_Y = size_y
        self.SCALING = scaling

        ############################
        # Turtle-drawing functions #
        ############################
        turtle.setup(self.SIZE_X*self.SCALING+50, self.SIZE_Y*self.SCALING+50)

        self.SCREEN = turtle.Screen()
        self.SCREEN.colormode(255)               # Set to RGB color mode

        self.TPEN = turtle.RawTurtle(self.SCREEN)
        self.TPEN.speed('fastest')               # Speed optimisations for turtle
        self.TPEN.ht()                           # Hide the little turtle that helps us draw

        if (sys.version_info.major > 2):    # Disable animations => Faster drawing of graphics
            self.SCREEN.tracer(0, 0)
        else:
            self.TPEN.tracer(0, 0)

    # (0, 0) adjusted to TOP-LEFT corner
    def setPos(self, pos):
        x = pos.x - self.SIZE_X/2
        y = -(pos.y -self.SIZE_Y/2)
        self.TPEN.up()
        self.TPEN.setpos(x*self.SCALING, y*self.SCALING)
        self.TPEN.down()

    def drawLine(self, s, e, c=None, thickness=1):
        self.TPEN.color('black' if c is None else c)
        self.TPEN.pensize(thickness)
        self.setPos(s)
        x = e.x - self.SIZE_X/2
        y = -(e.y -self.SIZE_Y/2)
        self.TPEN.setpos(x*self.SCALING, y*self.SCALING)

    # point(top-left) to point(bottom-right)
    def drawRect(self, p_tl, p_br, c=None, fill=None, thickness=1):
        self.TPEN.color('black' if c is None else c, 'white' if fill is None else fill)
        self.TPEN.pensize(thickness)
        self.setPos(p_tl)
        width = p_br.x - p_tl.x
        height = p_br.y - p_tl.y
        if (fill is not None):
            self.TPEN.begin_fill()
        self.TPEN.forward(width*self.SCALING)
        self.TPEN.right(90)
        self.TPEN.forward(height*self.SCALING)
        self.TPEN.right(90)
        self.TPEN.forward(width*self.SCALING)
        self.TPEN.right(90)
        self.TPEN.forward(height*self.SCALING)
        self.TPEN.right(90)
        if (fill is not None):
            self.TPEN.end_fill()

    def drawCircle(self, pos, radius, c=None, fill=None, thickness=1):
        self.TPEN.color('black' if c is None else c, 'white' if fill is None else fill)
        self.TPEN.pensize(thickness)
        self.setPos(pos)
        self.TPEN.up()
        self.TPEN.right(90)
        self.TPEN.forward(radius*self.SCALING)
        self.TPEN.left(90)
        self.TPEN.down()
        if (fill is not None):
            self.TPEN.begin_fill()
        self.TPEN.circle(radius*self.SCALING)
        if (fill is not None):
            self.TPEN.end_fill()
        self.TPEN.up()
        self.TPEN.left(90)
        self.TPEN.forward(radius*self.SCALING)
        self.TPEN.right(90)
        self.TPEN.down()

    def drawLabel(self, pos, txt, c=None, fontType="Arial", size=11, style="normal"):
        self.TPEN.color('black' if c is None else c)
        self.setPos(pos)
        self.TPEN.write(txt, align='center', font=(fontType, size, style))

    # Step function, complete this round of draw commands
    def STEP(self):
        turtle.update()

    ### BLOCKING ###
    # Delay Function
    def DELAY(self, t):
        turtle.update()
        time.sleep(t)

    ### BLOCKING ###
    # Pause Function
    def PAUSE(self):
        turtle.update()
        return raw_input("ENTER to continue (END to stop):")

    ### CLOSE TURTLE WINDOW ###
    def CLOSE(self):
        self.SCREEN.bye()

    ##################################
    # SIM SPECIFIC DRAWING FUNCTIONS #
    ##################################
    def t_BEGIN(self):      # Splash Screen!
        self.drawRect(Point(0, 0), Point(SIZE_X, SIZE_Y))
        self.drawLabel(Point(self.SIZE_X/2, self.SIZE_Y/2), "pyTurtle Graphics Lib", size=30, style="bold")

    def t_REFRESH(self):    # Refresh each turn
        self.TPEN.clear()
        self.drawRect(Point(0, 0), Point(self.SIZE_X, self.SIZE_Y))