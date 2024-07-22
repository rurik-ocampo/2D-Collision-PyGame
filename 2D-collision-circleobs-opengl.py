import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np

# Parameters
m1, m2 = 1.0, 1.0  # masses of the particles
r1, r2 = np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 0.0])  # initial positions
v1, v2 = np.array([1.0, 1.0, 0.0]), np.array([-1.0, -0.5, 0.0])  # initial velocities
particle_radius = 0.1  # particle radius

# Hollow circle obstacle parameters
obstacle_center = np.array([0.0, 0.0, 0.0])
inner_radius = 0.5
outer_radius = 0.8

# Time step
dt = 0.01

def detect_collision(r1, r2, radius):
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

def draw_hollow_circle(center, inner_radius, outer_radius):
    num_segments = 100
    theta = np.linspace(0, 2 * np.pi, num_segments)
    
    # Draw inner circle
    glBegin(GL_LINE_LOOP)
    for i in range(num_segments):
        x = inner_radius * np.cos(theta[i]) + center[0]
        y = inner_radius * np.sin(theta[i]) + center[1]
        glVertex3f(x, y, 0)
    glEnd()
    
    # Draw outer circle
    glBegin(GL_LINE_LOOP)
    for i in range(num_segments):
        x = outer_radius * np.cos(theta[i]) + center[0]
        y = outer_radius * np.sin(theta[i]) + center[1]
        glVertex3f(x, y, 0)
    glEnd()

def draw_sphere(position, radius, color):
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glColor3f(*color)
    glutSolidSphere(radius, 20, 20)
    glPopMatrix()

def main():
    pygame.init()
    display = (1280, 720)
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

        # Check for particle-particle collision and resolve it
        if detect_collision(r1, r2, particle_radius):
            v1, v2 = resolve_collision(r1, r2, v1, v2, m1, m2)

        # Check for particle-obstacle collision and resolve it
        if detect_collision(r1, obstacle_center, (particle_radius + outer_radius)*0.5):
            v1, _ = resolve_collision(r1, obstacle_center, v1, np.zeros(3), m1, np.inf)
        if detect_collision(r2, obstacle_center, (particle_radius + outer_radius)*0.5):
            v2, _ = resolve_collision(r2, obstacle_center, v2, np.zeros(3), m2, np.inf)

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw particles
        draw_sphere(r1, particle_radius, (1, 0, 0))
        draw_sphere(r2, particle_radius, (0, 0, 1))

        # Draw hollow circle obstacle
        glColor3f(0, 1, 0)
        draw_hollow_circle(obstacle_center, inner_radius, outer_radius)

        pygame.display.flip()
        pygame.time.wait(int(dt * 1000))

if __name__ == "__main__":
    main()
