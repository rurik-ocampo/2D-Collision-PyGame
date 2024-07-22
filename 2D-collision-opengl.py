import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np

# Parameters
m1, m2 = 1.0, 1.0  # masses of the particles
r1, r2 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])  # initial positions
v1, v2 = np.array([1.0, 1.0, 0.0]), np.array([-1.0, -1.0, 0.0])  # initial velocities
radius = 0.1  # particle radius

# Time step
dt = 0.01

def detect_collision(r1, r2):
    return np.linalg.norm(r1 - r2) < 2 * radius

def resolve_collision(r1, r2, v1, v2, m1, m2):
    # Relative position and velocity
    r_rel = r2 - r1
    v_rel = v2 - v1
    
    # Normal vector
    r_rel_norm = np.linalg.norm(r_rel)
    if r_rel_norm == 0:
        return v1, v2
    
    n = r_rel / r_rel_norm
    
    # Relative velocity in normal direction
    v_rel_n = np.dot(v_rel, n)
    
    # If velocities are separating, no collision
    if v_rel_n > 0:
        return v1, v2
    
    # Impulse scalar
    j = (2 * v_rel_n) / (1/m1 + 1/m2)
    
    # Update velocities
    v1_new = v1 + j * n / m1
    v2_new = v2 - j * n / m2
    
    return v1_new, v2_new

def draw_particle(position, color):
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glColor3f(*color)
    glutSolidSphere(radius, 20, 20)
    glPopMatrix()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    
    # Initialize GLUT
    glutInit(sys.argv)

    global r1, r2, v1, v2

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Update positions
        r1 += v1 * dt
        r2 += v2 * dt

        # Check for collision and resolve it
        if detect_collision(r1, r2):
            v1, v2 = resolve_collision(r1, r2, v1, v2, m1, m2)

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw particles
        draw_particle(r1, (1, 0, 0))
        draw_particle(r2, (0, 0, 1))

        pygame.display.flip()
        pygame.time.wait(int(dt * 1000))

if __name__ == "__main__":
    main()
