import pygame
import sys
import math
import numpy as np
import pyaudio
import threading
import queue

# Initialize pygame
pygame.init()
pygame.mixer.init()

# Screen dimensions
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dolphin Echolocation Pong")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
LIGHT_GRAY = (200, 200, 200)

# Game elements
BALL_RADIUS = 15
PADDLE_WIDTH, PADDLE_HEIGHT = 20, 120
PADDLE_SPEED = 8
BALL_SPEED_X, BALL_SPEED_Y = 6, 6

# Fonts
font_large = pygame.font.SysFont("consolas", 60)
font_medium = pygame.font.SysFont("consolas", 40)
font_small = pygame.font.SysFont("consolas", 28)

# Audio parameters for dolphin echolocation detection
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DETECTION_THRESHOLD = 0.5  # Sensitivity to clicks

# Create a queue to communicate between audio thread and main thread
audio_queue = queue.Queue()

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

class Paddle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.speed = PADDLE_SPEED
        self.score = 0
        
    def move(self, direction):
        if direction == "up":
            self.rect.y = max(0, self.rect.y - self.speed)
        elif direction == "down":
            self.rect.y = min(HEIGHT - self.rect.height, self.rect.y + self.speed)
            
    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self.rect)
        pygame.draw.rect(surface, LIGHT_GRAY, self.rect, 2)

class Ball:
    def __init__(self, x, y, radius):
        self.rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        self.radius = radius
        self.speed_x = BALL_SPEED_X
        self.speed_y = BALL_SPEED_Y
        self.active = False
        self.trail = []
        
    def move(self):
        if not self.active:
            return
            
        # Add current position to trail (limit trail length)
        self.trail.append((self.rect.centerx, self.rect.centery))
        if len(self.trail) > 10:
            self.trail.pop(0)
            
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        
        # Top and bottom collisions
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.speed_y *= -1
            
    def reset(self):
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.speed_x = BALL_SPEED_X * (1 if np.random.random() > 0.5 else -1)
        self.speed_y = BALL_SPEED_Y * (1 if np.random.random() > 0.5 else -1)
        self.trail = []
        self.active = False
        
    def draw(self, surface):
        # Draw trail
        for i, pos in enumerate(self.trail):
            radius = max(1, self.radius * i // len(self.trail))
            pygame.draw.circle(surface, GRAY, pos, radius)
        
        # Draw ball
        pygame.draw.circle(surface, WHITE, self.rect.center, self.radius)
        pygame.draw.circle(surface, LIGHT_GRAY, self.rect.center, self.radius - 5, 2)

# Create game objects
left_paddle = Paddle(50, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
right_paddle = Paddle(WIDTH - 50 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
ball = Ball(WIDTH // 2, HEIGHT // 2, BALL_RADIUS)

# Audio detection variables
detected_frequency = None
click_animation = 0
detection_history = []

def draw_ui(surface):
    # Draw scores
    left_score = font_large.render(str(left_paddle.score), True, WHITE)
    right_score = font_large.render(str(right_paddle.score), True, WHITE)
    surface.blit(left_score, (WIDTH // 4 - left_score.get_width() // 2, 30))
    surface.blit(right_score, (3 * WIDTH // 4 - right_score.get_width() // 2, 30))
    
    # Draw center line
    for y in range(0, HEIGHT, 30):
        pygame.draw.rect(surface, GRAY, (WIDTH // 2 - 5, y, 10, 15))
    
    # Draw instructions
    instructions = font_small.render("Left Paddle: W/S keys", True, WHITE)
    surface.blit(instructions, (100, HEIGHT - 40))
    
    instructions2 = font_small.render("Right Paddle: Dolphin clicks (50Hz = Down, 90Hz = Up)", True, WHITE)
    surface.blit(instructions2, (WIDTH - 550, HEIGHT - 40))
    
    # Draw echolocation indicator
    pygame.draw.rect(surface, GRAY, (WIDTH - 250, 30, 200, 40))
    status = font_small.render("Detection: ", True, WHITE)
    surface.blit(status, (WIDTH - 240, 38))
    
    if detected_frequency:
        freq_text = font_small.render(f"{detected_frequency}Hz", True, WHITE)
        surface.blit(freq_text, (WIDTH - 120, 38))
    
    # Draw click animation
    if click_animation > 0:
        pygame.draw.circle(surface, WHITE, (WIDTH - 120, 60), click_animation, 2)
    
    # Draw frequency visualization
    pygame.draw.rect(surface, GRAY, (WIDTH - 300, HEIGHT - 150, 280, 120))
    pygame.draw.rect(surface, BLACK, (WIDTH - 295, HEIGHT - 145, 270, 110))
    title = font_small.render("Frequency Detection", True, WHITE)
    surface.blit(title, (WIDTH - 290, HEIGHT - 145))
    
    # Draw history bars
    if detection_history:
        bar_width = 270 / len(detection_history)
        for i, freq in enumerate(detection_history):
            height = min(100, freq * 0.5) if freq > 0 else 0
            color = WHITE if freq == 50 or freq == 90 else GRAY
            pygame.draw.rect(surface, color, (
                WIDTH - 295 + i * bar_width,
                HEIGHT - 40 - height,
                bar_width - 2,
                height
            ))

def detect_echolocation():
    """Thread function to detect echolocation frequencies from microphone"""
    global detected_frequency, click_animation, detection_history
    
    while True:
        try:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Apply FFT
            fft_data = np.fft.rfft(audio_data)
            frequencies = np.fft.rfftfreq(len(audio_data), 1.0 / RATE)
            
            # Find magnitudes at target frequencies
            mag_50 = np.abs(fft_data[np.abs(frequencies - 50).argmin()])
            mag_90 = np.abs(fft_data[np.abs(frequencies - 90).argmin()])
            
            # Find the dominant frequency in the range 40-100Hz
            idx = np.where((frequencies >= 40) & (frequencies <= 100))[0]
            if len(idx) > 0:
                max_idx = idx[np.argmax(np.abs(fft_data[idx]))]
                dominant_freq = int(frequencies[max_idx])
                detection_history.append(dominant_freq)
                if len(detection_history) > 30:
                    detection_history.pop(0)
            
            # Threshold detection
            if mag_50 > CHUNK * DETECTION_THRESHOLD:
                detected_frequency = 50
                click_animation = 30
                audio_queue.put("down")
            elif mag_90 > CHUNK * DETECTION_THRESHOLD:
                detected_frequency = 90
                click_animation = 30
                audio_queue.put("up")
                
        except Exception as e:
            print(f"Audio error: {e}")
            break

# Start the audio detection thread
audio_thread = threading.Thread(target=detect_echolocation, daemon=True)
audio_thread.start()

# Game state
game_active = False
winner = None

# Main game loop
clock = pygame.time.Clock()
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stream.stop_stream()
            stream.close()
            p.terminate()
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if not game_active and winner is None:
                    game_active = True
                    ball.active = True
                elif winner:
                    # Reset game
                    left_paddle.score = 0
                    right_paddle.score = 0
                    ball.reset()
                    winner = None
                    game_active = True
                    ball.active = True
    
    # Process audio queue
    while not audio_queue.empty():
        command = audio_queue.get()
        if command == "up":
            right_paddle.move("up")
        elif command == "down":
            right_paddle.move("down")
    
    # Handle keyboard input for left paddle
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        left_paddle.move("up")
    if keys[pygame.K_s]:
        left_paddle.move("down")
    
    # Move ball
    ball.move()
    
    # Ball collisions with paddles
    if ball.rect.colliderect(left_paddle.rect):
        ball.speed_x = abs(ball.speed_x)
        # Add angle based on where the ball hit the paddle
        relative_y = (left_paddle.rect.centery - ball.rect.centery) / (PADDLE_HEIGHT / 2)
        ball.speed_y = -relative_y * BALL_SPEED_Y
        
    if ball.rect.colliderect(right_paddle.rect):
        ball.speed_x = -abs(ball.speed_x)
        relative_y = (right_paddle.rect.centery - ball.rect.centery) / (PADDLE_HEIGHT / 2)
        ball.speed_y = -relative_y * BALL_SPEED_Y
    
    # Scoring
    if ball.rect.left <= 0:
        right_paddle.score += 1
        ball.reset()
        game_active = False
        if right_paddle.score >= 5:
            winner = "Dolphin"
    
    if ball.rect.right >= WIDTH:
        left_paddle.score += 1
        ball.reset()
        game_active = False
        if left_paddle.score >= 5:
            winner = "Player"
    
    # Update animation
    if click_animation > 0:
        click_animation -= 1
    
    # Drawing
    screen.fill(BLACK)
    
    # Draw game elements
    left_paddle.draw(screen)
    right_paddle.draw(screen)
    ball.draw(screen)
    draw_ui(screen)
    
    # Draw game messages
    if not game_active:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))
        
        if winner:
            winner_text = font_large.render(f"{winner} Wins!", True, WHITE)
            screen.blit(winner_text, (WIDTH // 2 - winner_text.get_width() // 2, HEIGHT // 2 - 50))
            restart_text = font_medium.render("Press SPACE to Play Again", True, WHITE)
            screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 50))
        else:
            start_text = font_large.render("DOLPHIN ECHOLOCATION PONG", True, WHITE)
            screen.blit(start_text, (WIDTH // 2 - start_text.get_width() // 2, HEIGHT // 2 - 100))
            
            freq_text = font_medium.render("Use 50Hz clicks for DOWN, 90Hz for UP", True, LIGHT_GRAY)
            screen.blit(freq_text, (WIDTH // 2 - freq_text.get_width() // 2, HEIGHT // 2))
            
            start_help = font_medium.render("Press SPACE to Start", True, WHITE)
            screen.blit(start_help, (WIDTH // 2 - start_help.get_width() // 2, HEIGHT // 2 + 100))
    
    pygame.display.flip()
    clock.tick(60)
