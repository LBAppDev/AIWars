import pygame
import sys

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class Square:
    def __init__(self, color, x, y, width, height):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.velocity = 5

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height))

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

def main():
    pygame.init()
    screen_width, screen_height = 500, 500
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Two Squares Game")
    clock = pygame.time.Clock()

    player1 = Square(RED, 50, 50, 50, 50)
    player2 = Square(BLUE, 400, 400, 50, 50)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            player1.move(0, -player1.velocity)
        if keys[pygame.K_s]:
            player1.move(0, player1.velocity)
        if keys[pygame.K_a]:
            player1.move(-player1.velocity, 0)
        if keys[pygame.K_d]:
            player1.move(player1.velocity, 0)

        if keys[pygame.K_UP]:
            player2.move(0, -player2.velocity)
        if keys[pygame.K_DOWN]:
            player2.move(0, player2.velocity)
        if keys[pygame.K_LEFT]:
            player2.move(-player2.velocity, 0)
        if keys[pygame.K_RIGHT]:
            player2.move(player2.velocity, 0)

        # Drawing
        screen.fill(WHITE)
        player1.draw(screen)
        player2.draw(screen)
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
