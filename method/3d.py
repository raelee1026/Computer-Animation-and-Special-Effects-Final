import pygame
import numpy as np
import math
import random
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

# 初始化
pygame.init()
WIDTH, HEIGHT = 1200, 800
pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Hair Physics Simulation - Improved")

# 設置OpenGL
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glEnable(GL_LINE_SMOOTH)
glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

# 設置透視投影
gluPerspective(45, (WIDTH/HEIGHT), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        length = self.length()
        if length > 0:
            return Vector3(self.x/length, self.y/length, self.z/length)
        return Vector3(0, 0, 0)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def to_list(self):
        return [self.x, self.y, self.z]

class HairStrand3D:
    def __init__(self, start_pos, initial_direction, length, segments=15, head_radius=0.8):
        self.segments = segments
        self.segment_length = length / segments
        self.positions = []
        self.old_positions = []
        self.constraints = []
        self.head_radius = head_radius
        self.head_center = Vector3(0, 0, 0)
        
        # 初始化髮絲位置 - 沿著初始方向生長
        for i in range(segments + 1):
            # 添加一些隨機性讓頭髮看起來更自然
            random_offset = Vector3(
                random.uniform(-0.015, 0.015),
                random.uniform(-0.008, 0.008), 
                random.uniform(-0.015, 0.015)
            )
            
            pos = start_pos + initial_direction * (i * self.segment_length) + random_offset
            self.positions.append(pos)
            self.old_positions.append(Vector3(pos.x, pos.y, pos.z))
        
        # 創建約束連接
        for i in range(segments):
            self.constraints.append((i, i + 1, self.segment_length))
    
    def apply_sphere_collision(self, position, safety_margin=0.05):
        """確保髮絲點不會進入頭部球體內部"""
        to_center = position - self.head_center
        distance = to_center.length()
        min_distance = self.head_radius + safety_margin
        
        if distance < min_distance and distance > 0:
            # 將點推到球體表面外
            correction = to_center.normalize() * min_distance
            return self.head_center + correction
        return position
    
    def update(self, wind, gravity=Vector3(0, -0.01, 0), damping=0.99):
        # Verlet積分更新位置
        for i in range(1, len(self.positions)):  # 跳過根部（固定點）
            pos = self.positions[i]
            old_pos = self.old_positions[i]
            
            # 計算速度
            velocity = (pos - old_pos) * damping
            
            # 保存當前位置
            self.old_positions[i] = Vector3(pos.x, pos.y, pos.z)
            
            # 應用力（重力 + 風力）
            # 風力隨髮絲長度增強
            wind_effect = wind * (0.03 + i * 0.008)
            
            new_pos = pos + velocity + gravity + wind_effect
            
            # 應用球體碰撞檢測
            self.positions[i] = self.apply_sphere_collision(new_pos)
    
    def satisfy_constraints(self, iterations=3):
        # 多次迭代滿足約束條件
        for _ in range(iterations):
            for i, j, target_length in self.constraints:
                pos1 = self.positions[i]
                pos2 = self.positions[j]
                
                # 計算當前距離向量
                diff = pos2 - pos1
                distance = diff.length()
                
                if distance > 0:
                    # 計算修正量
                    difference = target_length - distance
                    correction = diff.normalize() * (difference * 0.5)
                    
                    # 只移動非根部點
                    if i > 0:
                        new_pos1 = self.positions[i] - correction
                        self.positions[i] = self.apply_sphere_collision(new_pos1)
                    if j > 0:
                        new_pos2 = self.positions[j] + correction
                        self.positions[j] = self.apply_sphere_collision(new_pos2)
    
    def draw(self, color=(0.4, 0.2, 0.1, 1.0)):
        glColor4f(*color)
        glLineWidth(1.5)
        
        glBegin(GL_LINE_STRIP)
        for pos in self.positions:
            glVertex3f(pos.x, pos.y, pos.z)
        glEnd()

class Head3D:
    def __init__(self, radius=1.0):
        self.radius = radius
        self.position = Vector3(0, 0, 0)
    
    def draw(self):
        # 繪製頭部（球體）
        glColor3f(0.9, 0.7, 0.6)  # 膚色
        
        glPushMatrix()
        glTranslatef(self.position.x, self.position.y, self.position.z)
        
        # 使用GLU繪製球體
        quadric = gluNewQuadric()
        gluSphere(quadric, self.radius, 32, 32)
        gluDeleteQuadric(quadric)
        
        # 繪製眼睛
        # glColor3f(0.0, 0.0, 0.0)
        # glPushMatrix()
        # glTranslatef(-0.3, 0.2, 0.8)
        # eye_quadric = gluNewQuadric()
        # gluSphere(eye_quadric, 0.08, 16, 16)
        # gluDeleteQuadric(eye_quadric)
        # glPopMatrix()
        
        # glPushMatrix()
        # glTranslatef(0.3, 0.2, 0.8)
        # eye_quadric2 = gluNewQuadric()
        # gluSphere(eye_quadric2, 0.08, 16, 16)
        # gluDeleteQuadric(eye_quadric2)
        # glPopMatrix()
        
        # 繪製鼻子
        # glColor3f(0.8, 0.6, 0.5)
        # glPushMatrix()
        # glTranslatef(0.0, -0.1, 0.9)
        # nose_quadric = gluNewQuadric()
        # gluSphere(nose_quadric, 0.06, 16, 16)
        # gluDeleteQuadric(nose_quadric)
        # glPopMatrix()
        
        glPopMatrix()

class HairSystem3D:
    def __init__(self, head_radius=0.8):
        self.head_radius = head_radius
        self.head = Head3D(head_radius)
        self.strands = []
        self.wind = Vector3(0, 0, 0)
        self.hair_colors = {
            1: (0.4, 0.2, 0.1, 1.0),  # 棕色
            2: (0.8, 0.7, 0.3, 1.0),  # 金色
            3: (0.1, 0.1, 0.1, 1.0),  # 黑色
            4: (0.6, 0.2, 0.2, 1.0),  # 紅色
            5: (0.3, 0.3, 0.3, 1.0),  # 灰色
        }
        self.current_color = 1
        
        self.create_hair()
    
    def create_hair(self):
        # 大幅增加髮絲數量
        num_strands = 800  # 從200增加到800
        
        # 創建多層頭髮以增加密度
        hair_layers = [
            {'count': 300, 'radius_offset': 0.02, 'length_range': (0.7, 1.1)},  # 外層
            {'count': 300, 'radius_offset': 0.01, 'length_range': (0.6, 1.0)},  # 中層
            {'count': 200, 'radius_offset': 0.005, 'length_range': (0.5, 0.9)}, # 內層
        ]
        
        for layer in hair_layers:
            for i in range(layer['count']):
                # 生成頭髮的角度範圍 - 改進分佈
                phi = random.uniform(0, 2 * math.pi)  # 水平角度 0-360度
                
                # 更自然的垂直分佈
                # 使用更複雜的分佈來避免頭髮在臉部和脖子生長
                u = random.random()
                # 這個分佈讓更多頭髮在頭頂，較少在側面
                theta = math.acos(1 - u * 0.7)  # 0到約130度
                
                # 排除臉部區域（前方下半部）
                if theta > math.pi * 0.4 and abs(phi) < math.pi * 0.4:
                    continue  # 跳過臉部區域
                
                # 計算頭皮上的位置
                x = math.sin(theta) * math.cos(phi)
                y = math.cos(theta)  # y軸向上
                z = math.sin(theta) * math.sin(phi)
                
                # 髮根位置（稍微向外偏移以避免穿透頭部）
                root_pos = Vector3(x, y, z) * (self.head_radius + layer['radius_offset'])
                
                # 計算初始生長方向（更自然的方向）
                direction = Vector3(x, y, z).normalize()
                
                # 根據位置調整初始方向
                if y > 0.6:  # 頭頂的頭髮
                    # 頭頂頭髮向外並向下
                    direction.y -= random.uniform(0.2, 0.4)
                elif y > 0.2:  # 側面頭髮
                    # 側面頭髮稍微向外和向下
                    direction = direction * 0.8 + Vector3(x * 0.2, -0.3, z * 0.2)
                else:  # 後腦和側後方
                    # 向外並稍微向下
                    direction.y -= random.uniform(0.1, 0.3)
                
                # 添加隨機變化讓頭髮方向更自然
                direction.x += random.uniform(-0.1, 0.1)
                direction.z += random.uniform(-0.1, 0.1)
                direction = direction.normalize()
                
                # 髮絲長度變化
                min_length, max_length = layer['length_range']
                if y > 0.7:  # 頭頂頭髮
                    hair_length = random.uniform(min_length * 0.8, max_length * 0.9)
                elif y > 0.3:  # 側面頭髮
                    hair_length = random.uniform(min_length, max_length)
                else:  # 後腦頭髮
                    hair_length = random.uniform(min_length * 0.9, max_length * 1.1)
                
                # 根據層次調整分段數
                segments = random.randint(10, 15)
                
                strand = HairStrand3D(root_pos, direction, hair_length, segments, self.head_radius)
                self.strands.append(strand)
        
        print(f"Created {len(self.strands)} hair strands")
    
    def set_wind(self, wind_vector):
        self.wind = wind_vector
    
    def update(self):
        for strand in self.strands:
            strand.update(self.wind)
            strand.satisfy_constraints()
    
    def draw(self):
        # 繪製頭部
        self.head.draw()
        
        # 繪製頭髮 - 分批渲染以提高性能
        hair_color = self.hair_colors[self.current_color]
        
        # 為了視覺效果，給不同長度的頭髮稍微不同的顏色
        for i, strand in enumerate(self.strands):
            # 添加輕微的顏色變化
            color_variation = 1.0 + (i % 10 - 5) * 0.02
            varied_color = (
                min(1.0, hair_color[0] * color_variation),
                min(1.0, hair_color[1] * color_variation),
                min(1.0, hair_color[2] * color_variation),
                hair_color[3]
            )
            strand.draw(varied_color)
    
    def change_hair_color(self, color_index):
        if color_index in self.hair_colors:
            self.current_color = color_index

def main():
    clock = pygame.time.Clock()
    hair_system = HairSystem3D()
    
    # 旋轉角度
    rotation_x = 0
    rotation_y = 0
    
    # 風力參數
    wind_strength = 0.02
    wind_x, wind_y, wind_z = 0, 0, 0
    
    running = True
    mouse_dragging = False
    last_mouse_pos = (0, 0)
    
    print("3D Hair Physics Simulation - Improved Version")
    print("Controls:")
    print("Mouse: Rotate view")
    print("WASD: Control wind (X/Z axis)")
    print("Q/E: Control wind (Y axis)")
    print("1-5: Change hair color")
    print("R: Reset wind")
    print("Space: Random wind burst")
    print("ESC: Exit")
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        # 處理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    wind_x = wind_y = wind_z = 0
                elif event.key == pygame.K_SPACE:
                    wind_x = random.uniform(-0.06, 0.06)
                    wind_y = random.uniform(-0.03, 0.03)
                    wind_z = random.uniform(-0.06, 0.06)
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    color_index = event.key - pygame.K_0
                    hair_system.change_hair_color(color_index)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左鍵
                    mouse_dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    mouse_pos = pygame.mouse.get_pos()
                    dx = mouse_pos[0] - last_mouse_pos[0]
                    dy = mouse_pos[1] - last_mouse_pos[1]
                    rotation_y += dx * 0.5
                    rotation_x += dy * 0.5
                    # 限制垂直旋轉角度
                    rotation_x = max(-80, min(80, rotation_x))
                    last_mouse_pos = mouse_pos
        
        # 處理持續按鍵
        keys = pygame.key.get_pressed()
        
        # 風力控制
        if keys[pygame.K_a]:
            wind_x = max(wind_x - wind_strength, -0.1)
        if keys[pygame.K_d]:
            wind_x = min(wind_x + wind_strength, 0.1)
        if keys[pygame.K_w]:
            wind_z = max(wind_z - wind_strength, -0.1)
        if keys[pygame.K_s]:
            wind_z = min(wind_z + wind_strength, 0.1)
        if keys[pygame.K_q]:
            wind_y = min(wind_y + wind_strength, 0.05)
        if keys[pygame.K_e]:
            wind_y = max(wind_y - wind_strength, -0.05)
        
        # 風力衰減
        wind_x *= 0.985
        wind_y *= 0.985
        wind_z *= 0.985
        
        # 更新髮絲物理
        hair_system.set_wind(Vector3(wind_x, wind_y, wind_z))
        hair_system.update()
        
        # 渲染
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glPushMatrix()
        
        # 應用旋轉
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)
        
        # 繪製場景
        hair_system.draw()
        
        glPopMatrix()
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()