import pygame
import sys
import math
import random

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)


class Bullet:
    def __init__(self, x, y, angle, velocity):
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = velocity
        self.width = 5
        self.height = 5

    def move(self):
        self.x += self.velocity * math.cos(math.radians(self.angle))
        self.y += self.velocity * math.sin(math.radians(self.angle))

    def draw(self, surface):
        pygame.draw.rect(surface, BLACK, (self.x, self.y, self.width, self.height))


class Square:
    def __init__(self, color, x, y, width, height, screen_width, screen_height, obstacles):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.velocity = 5
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.back_width = 10
        self.back_height = 10
        self.ray_length = 300
        self.angle = 0  # Initial angle
        self.health = 10  # Initial health
        self.bullets = []
        self.last_shot_time = pygame.time.get_ticks()  # Store the time when the last bullet was shot
        self.obstacles = obstacles

    def draw(self, surface):
        # Draw main square
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height))

        # Calculate the position of the back side (small square)
        back_center_x = self.x + self.width / 2 + self.width / 2 * math.cos(math.radians(self.angle))
        back_center_y = self.y + self.height / 2 + self.height / 2 * math.sin(math.radians(self.angle))

        # Draw back side
        pygame.draw.rect(surface, BLACK, (
        back_center_x - self.back_width / 2, back_center_y - self.back_height / 2, self.back_width, self.back_height))

        # Draw the direction indicator (ray)
        end_x = back_center_x + self.ray_length * math.cos(math.radians(self.angle))
        end_y = back_center_y + self.ray_length * math.sin(math.radians(self.angle))
        pygame.draw.line(surface, (255, 255, 0), (back_center_x, back_center_y), (end_x, end_y), 2)

        # Draw health
        font = pygame.font.Font(None, 24)
        text_surface = font.render(str(self.health), True, WHITE)
        text_rect = text_surface.get_rect(center=(self.x + self.width / 2, self.y + self.height / 2))
        surface.blit(text_surface, text_rect)

        # Draw bullets
        for bullet in self.bullets:
            bullet.draw(surface)

        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(surface, BLACK, obstacle)

    def move(self, dx, dy):
        new_x = self.x + dx
        new_y = self.y + dy

        # Check if the new position is within the screen boundaries and not colliding with obstacles
        if 0 <= new_x <= self.screen_width - self.width and not self.collides_with_obstacle(new_x, self.y):
            self.x = new_x
        if 0 <= new_y <= self.screen_height - self.height and not self.collides_with_obstacle(self.x, new_y):
            self.y = new_y

    def rotate_clockwise(self):
        self.angle -= 5  # Rotate 5 degrees clockwise

    def rotate_counterclockwise(self):
        self.angle += 5  # Rotate 5 degrees counterclockwise

    def shoot(self):
        # Check if enough time has passed since the last shot
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time >= 1000:  # 1000 milliseconds = 1 second
            bullet = Bullet(self.x + self.width / 2, self.y + self.height / 2, self.angle, 10)
            self.bullets.append(bullet)
            self.last_shot_time = current_time  # Update the last shot time

    def update_bullets(self):
        for bullet in self.bullets:
            bullet.move()
            # Remove bullets that go out of screen
            if bullet.x < 0 or bullet.x > self.screen_width or bullet.y < 0 or bullet.y > self.screen_height:
                self.bullets.remove(bullet)

            # Check for collision with obstacles
            for obstacle in self.obstacles:
                if (bullet.x >= obstacle[0] and bullet.x <= obstacle[0] + obstacle[2] and
                        bullet.y >= obstacle[1] and bullet.y <= obstacle[1] + obstacle[3]):
                    self.bullets.remove(bullet)
                    break

    def check_collision(self, other_square):
        for bullet in other_square.bullets:
            if (self.x < bullet.x < self.x + self.width) and (self.y < bullet.y < self.y + self.height):
                self.health -= 1
                other_square.bullets.remove(bullet)

    def collides_with_obstacle(self, x, y):
        # Check if the position collides with any obstacle
        for obstacle in self.obstacles:
            if (x < obstacle[0] + obstacle[2] and x + self.width > obstacle[0] and
                    y < obstacle[1] + obstacle[3] and y + self.height > obstacle[1]):
                return True
        return False


screen_width, screen_height = 500, 500


def generate_obstacles():
    num_obstacles = random.randint(30, 30)  # Random number of obstacles
    min_obstacle_size = 25
    max_obstacle_size = 30
    obstacles = []

    for _ in range(num_obstacles):
        obstacle_width = random.randint(min_obstacle_size, max_obstacle_size)
        obstacle_height = random.randint(min_obstacle_size, max_obstacle_size)
        obstacle_x = random.randint(0, screen_width - obstacle_width)
        obstacle_y = random.randint(0, screen_height - obstacle_height)

        # Add obstacle to the list
        obstacles.append((obstacle_x, obstacle_y, obstacle_width, obstacle_height))
    return obstacles


def generate_obstacles_static():
    x = [120, 200, 350]
    y = [50, 200, 290]
    width = [20, 100, 20]
    height = [150, 100, 150]
    obstacles = []
    for i in range(len(x)):
        obstacles.append((x[i], y[i], width[i], height[i]))

    return obstacles


def main():
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Two Squares Game")
    clock = pygame.time.Clock()

    obstacles = generate_obstacles_static()

    player1 = Square(RED, 50, 50, 50, 50, screen_width, screen_height, obstacles)
    player2 = Square(BLUE, 400, 400, 50, 50, screen_width, screen_height, obstacles)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Control for player 1
        if keys[pygame.K_w]:
            player1.move(0, -player1.velocity)
        if keys[pygame.K_s]:
            player1.move(0, player1.velocity)
        if keys[pygame.K_a]:
            player1.move(-player1.velocity, 0)
        if keys[pygame.K_d]:
            player1.move(player1.velocity, 0)
        if keys[pygame.K_e]:
            player1.rotate_clockwise()
        if keys[pygame.K_q]:
            player1.rotate_counterclockwise()
        if keys[pygame.K_r]:
            player1.shoot()

        # Control for player 2
        if keys[pygame.K_UP]:
            player2.move(0, -player2.velocity)
        if keys[pygame.K_DOWN]:
            player2.move(0, player2.velocity)
        if keys[pygame.K_LEFT]:
            player2.move(-player2.velocity, 0)
        if keys[pygame.K_RIGHT]:
            player2.move(player2.velocity, 0)
        if keys[pygame.K_i]:
            player2.rotate_clockwise()
        if keys[pygame.K_p]:
            player2.rotate_counterclockwise()
        if keys[pygame.K_u]:
            player2.shoot()

        # Update bullets
        player1.update_bullets()
        player2.update_bullets()

        # Check for collisions and decrement health
        player1.check_collision(player2)
        player2.check_collision(player1)

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
