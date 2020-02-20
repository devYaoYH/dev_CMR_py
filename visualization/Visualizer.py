from .turtle_engine import Point, Vector, Turtle
import numpy as np
import colorsys

class Visualizer(object):

	def __init__(self, width=600, height=400, scaling=1):
		self.width = width
		self.height = height
		self.padding = 15
		# Initialize turtle canvas
		self.canvas = Turtle(size_x=self.width, size_y=self.height, scaling=scaling)
		self.layers = []

	class Layer(object):
		def __init__(self, vis, tl_pos, cell_size, data):
			self.vis_callback = vis
			self.anchor = tl_pos
			self.data = data
			self.size = cell_size
			self.width = self.size*len(self.data[0])
			self.height = self.size*len(self.data)

		def update(self, data):
			self.data = data
			# self.vis_callback.update()

	# data_size: tuple containing shape of data
	def add_layer(self, data_size):
		# Compute where next layer should go
		# Top-Left corner as anchor point for data
		l_pos = Point(self.padding + sum([l.width + self.padding for l in self.layers]), self.padding)
		# Size of each cell (scales with data_size)
		c_size = min((self.height - 2*self.padding)/data_size[0], (self.width - 2*self.padding)/data_size[1])

		# Add a Layer object to our list of layers
		layer = self.Layer(self, l_pos, c_size, np.zeros(data_size))
		self.layers.append(layer)

		# Return handle to layer
		return layer

	def update(self):
		# Update canvas with changes
		self.canvas.t_REFRESH()
		for l in self.layers:
			anchor = l.anchor
			l_data = l.data
			for r in range(len(l_data)):
				r_y = anchor.y + r*l.size
				for c in range(len(l_data[r])):
					c_x = anchor.x + c*l.size
					self.canvas.drawRect(Point(c_x, r_y), Point(c_x + l.size, r_y + l.size), fill=self.get_fill(l_data[r,c]))
		self.canvas.STEP()

	def get_fill(self, f_val):
		CLAMP = 2
		# Clamp f_val within range [-1, 1]
		f_val += CLAMP/2
		if (f_val < 0):
			f_val = 0
		elif (f_val > CLAMP):
			f_val = CLAMP
		hue = f_val/CLAMP
		sat = 1
		val = 1
		rgb_col = colorsys.hsv_to_rgb(hue, sat, val)
		r, g, b = rgb_col
		return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

	def pause(self):
		self.canvas.PAUSE()

	def exit(self):
		self.canvas.CLOSE()