from matplotlib import pyplot as plt
import numpy as np
from generator import ImageGenerator
from pattern import Checker, Circle, Spectrum

if __name__ == '__main__':
    
    """check = Checker(250,25)
    check.draw()
    check.show()
    """
    circle = Circle(500, 50, (100,100))
    circle.draw() 
    circle.show()

    """
    spectrum = Spectrum(500)
    spectrum.draw() 
    spectrum.show()
    
    img_gen = ImageGenerator(file_path='exercise_data', label_path='Labels.json', batch_size=12, 
                    image_size=[32,32,3], rotation=True, mirroring=True, shuffle=True)

    img_gen.show()
    img_gen.next()
    img_gen.show()
    img_gen.next()
    img_gen.show()
    """