import matplotlib.pyplot as plt
import numpy as np

class Checker:

    def __init__(self, resolution: int, tile_size: int):
        
        # Assert that the resolution is divisible by 2*tile_size
        assert resolution % (2*tile_size) == 0

        # Attr assigment
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None
    
    def draw(self):
        
        # Create a 2*tile_size x 2*tile_size grid --> Ex [0, 0]
        #                                                [0, 0]
        check = np.zeros((2*self.tile_size, 2*self.tile_size))
        
        # Assign bottom left part of the grid as a 1 --> Ex [0, 0]
        #                                                   [1, 0]
        check[self.tile_size:, :self.tile_size] = 1
        
        # The other way around, top right part of the grid --> Ex [0, 1]
        #                                                         [1, 0]    
        check[:self.tile_size, self.tile_size:] = 1

        # Tile the created grid resolution / 2*tile_size times.
        # For example, in a 250, 25 resolution, tile_size checker --> 250/(2*25) = 5
        checker = np.tile(check, (int(self.resolution/(2*self.tile_size)), int(self.resolution/(2*self.tile_size))))

        # IMPORTANT, COPY THE RESULT TO AVOID MEMORY LEAKS!
        self.output = np.copy(checker)

        return np.copy(checker)
        
        
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Circle:

    def __init__(self, resolution: int, radius: int, position):
        # Attr assignment
        self.resolution = resolution
        self.radius = radius
        self.position = position

        self.output = None

    def draw(self):
        
        # Create a grid where x_cir and y_cir are the x and y coordinates of the grid, respectively
        x_cir, y_cir = np.meshgrid(range(self.resolution), range(self.resolution))
        
        # Better handling the position of the circle
        x_pos, y_pos = self.position
        
        # The circle equation: (x-a)^2 + (y-b)^2 = r^2
        # This creates in equation a faded image, where anything inside the circle radius is 0
        equation = (x_cir - x_pos)**2 + (y_cir - y_pos)**2 #- self.radius**2
        
        # Check which values are below the radius value, and convert them to integers
        circle = (equation < self.radius**2)

        # Copy of the circle
        self.output = np.copy(circle)

        return np.copy(circle)
    
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Spectrum:

    def __init__(self, resolution: int):
        self.resolution = resolution
        self.output = None

    def draw(self):
        
        # Create a (res,res,3) matrix for the image
        spectrum = np.zeros((self.resolution, self.resolution, 3))
        # Fill red channel with a right -> left descending matrix [1,   ..., 0.9]
        #                                                         [0.9, ..., 0.8]
        spectrum[:,:,0] = np.linspace(0.0,1.0,self.resolution**2).reshape(self.resolution, self.resolution)
        # Same for green, but transpose the matrix
        spectrum[:,:,1] = np.linspace(1.0,0.0,self.resolution**2).reshape(self.resolution, self.resolution).T
        # Blue is the flipped version of red
        spectrum[:,:,2] = np.linspace(1.0,0.0,self.resolution**2).reshape(self.resolution, self.resolution)
        
        # Rotate to put the image as example
        spectrum = np.rot90(spectrum)

        self.output = np.copy(spectrum)

        return np.copy(spectrum)
    
    def show(self):
        plt.imshow(self.output)
        plt.show()