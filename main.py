import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygame
import pygame.gfxdraw
import sys
import os
import math

#CNN & class names

cnn_model = load_model('hand_drawing_model_Inception.h5')

class_names = [
    'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'airplane',
    'alarm clock', 'apple', 'ball', 'banana', 'basket', 'bat', 'bed', 'bee',
    'bicycle', 'bird', 'book', 'bracelet', 'broom', 'bus', 'butterfly',
    'cactus', 'cake', 'calculator', 'camel', 'camera', 'candle', 'cannon',
    'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone', 'chair',
    'church', 'clock', 'cloud', 'cookie', 'couch', 'cow', 'cup', 'dolphin',
    'door', 'drums', 'elephant', 'eyeglasses', 'fish', 'floor lamp',
    'flower', 'giraffe', 'guitar', 'hammer', 'hat', 'headphones',
    'helicopter', 'helmet', 'house', 'ice cream', 'laptop', 'lion',
    'mermaid', 'monkey', 'octopus', 'panda', 'pants', 'pencil', 'penguin',
    'pineapple', 'pizza', 'rabbit', 'sailboat', 'sandwich', 'sea turtle',
    'sheep', 'shoe', 'smiley face', 'snowman', 'spider', 'squirrel', 'star',
    'strawberry', 'sun', 't-shirt', 'television', 'traffic light', 'train',
    'tree', 'umbrella', 'vase', 'watermelon'
]

# preprocessing & predicting

def preprocess_canvas_np(canvas_np):
    gray = cv2.cvtColor(canvas_np, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (128, 128))
    img_input = np.expand_dims(resized, axis=0)
    return img_input


def predict_canvas(canvas_np):
    img_input = preprocess_canvas_np(canvas_np)
    prediction = cnn_model.predict(img_input, verbose=0)
    confidence = float(prediction[0].max())
    class_idx = int(np.argmax(prediction[0]))
    if class_names[class_idx] == 'fish' and confidence < 0.15:
        return 'NONE', confidence
    return class_names[class_idx], confidence

#palette

BG           = (13,  13,  20)
PANEL_BG     = (20,  22,  32)
PANEL_BORDER = (40,  44,  62)
ACCENT       = (90, 160, 255)
TEXT_PRI     = (230, 232, 245)
TEXT_SEC     = (120, 124, 155)
TEXT_DIM     = ( 60,  64,  90)
WHITE        = (255, 255, 255)
BLACK        = (  0,   0,   0)
GREEN        = ( 80, 220, 130)
ORANGE       = (255, 165,  70)


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def rounded_rect(surf, color, rect, r, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=r)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=r)

EXAMPLE_SIZE = 180
_example_cache: dict = {}


def load_example(class_name: str) -> pygame.Surface:
    if class_name in _example_cache:
        return _example_cache[class_name]
    safe = class_name.lower().replace(' ', '_').replace('/', '_')
    for ext in ('png', 'jpg', 'jpeg', 'webp', 'jfif'):
        path = os.path.join('examples', f'{safe}.{ext}')
        if os.path.isfile(path):
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.smoothscale(img, (EXAMPLE_SIZE, EXAMPLE_SIZE))
            _example_cache[class_name] = img
            return img
    ph = pygame.Surface((EXAMPLE_SIZE, EXAMPLE_SIZE), pygame.SRCALPHA)
    ph.fill((30, 32, 45))
    rounded_rect(ph, (40, 44, 62), (0, 0, EXAMPLE_SIZE, EXAMPLE_SIZE), 12, 1, (60, 65, 90))
    try:
        f = pygame.font.SysFont('Segoe UI', 13)
        t = f.render('no preview', True, (70, 74, 100))
        ph.blit(t, (EXAMPLE_SIZE // 2 - t.get_width() // 2, EXAMPLE_SIZE // 2 - t.get_height() // 2))
    except Exception:
        pass
    _example_cache[class_name] = ph
    return ph

#  Slider

class Slider:
    def __init__(self, x, y, w, min_val, max_val, init_val, label=''):
        self.rect = pygame.Rect(x, y, w, 6)
        self.min = min_val
        self.max = max_val
        self.value = float(init_val)
        self.label = label
        self.dragging = False
        self.handle_r = 10

    def _frac(self):
        return (self.value - self.min) / (self.max - self.min)

    def handle_pos(self):
        fx = self.rect.x + int(self._frac() * self.rect.w)
        return fx, self.rect.centery

    def draw(self, surf, font_sm):
        cy = self.rect.centery
        rounded_rect(surf, (35, 38, 55), pygame.Rect(self.rect.x, cy - 3, self.rect.w, 6), 3)
        fw = int(self._frac() * self.rect.w)
        if fw > 0:
            rounded_rect(surf, ACCENT, pygame.Rect(self.rect.x, cy - 3, fw, 6), 3)
        hx, hy = self.handle_pos()
        pygame.draw.circle(surf, PANEL_BG, (hx, hy), self.handle_r)
        pygame.draw.circle(surf, ACCENT, (hx, hy), self.handle_r, 2)
        pygame.draw.circle(surf, ACCENT, (hx, hy), 4)
        if self.label:
            lbl = font_sm.render(self.label, True, TEXT_SEC)
            surf.blit(lbl, (self.rect.x, self.rect.y - 22))
        val_txt = font_sm.render(str(int(self.value)), True, TEXT_PRI)
        surf.blit(val_txt, (self.rect.right - val_txt.get_width(), self.rect.y - 22))

    def handle_event(self, event):
        hx, hy = self.handle_pos()
        hr = self.handle_r + 4
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if math.hypot(event.pos[0] - hx, event.pos[1] - hy) < hr:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rx = max(self.rect.x, min(self.rect.right, event.pos[0]))
            frac = (rx - self.rect.x) / self.rect.w
            self.value = self.min + frac * (self.max - self.min)
        return self.dragging

#  Button

class Button:
    def __init__(self, x, y, w, h, text, color=None, text_color=WHITE):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.base_color = color or ACCENT
        self.color = self.base_color
        self.text_color = text_color
        self.hovered = False
        self.pressed = False

    def draw(self, surf, font):
        c = lerp_color(self.color, WHITE, 0.12 if self.hovered else 0)
        c = lerp_color(c, BLACK, 0.2 if self.pressed else 0)
        rounded_rect(surf, c, self.rect, 8)
        lbl = font.render(self.text, True, self.text_color)
        surf.blit(lbl, lbl.get_rect(center=self.rect.center))

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                return True
        if event.type == pygame.MOUSEBUTTONUP:
            self.pressed = False
        return False


# Toggle Switch (pen / eraser)

class ToggleSwitch:
    def __init__(self, x, y, label_left='Pen', label_right='Eraser'):
        self.x = x
        self.y = y
        self.w = 52
        self.h = 26
        self.label_left  = label_left
        self.label_right = label_right
        self.is_right = False
        self.knob_anim = 0.0

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.w, self.h)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_right = not self.is_right
                return True
        return False

    def update(self, dt):
        target = 1.0 if self.is_right else 0.0
        self.knob_anim += (target - self.knob_anim) * min(1.0, dt * 14)

    def draw(self, surf, font_sm):
        track_color = lerp_color((50, 130, 220), (200, 80, 80), self.knob_anim)
        rounded_rect(surf, track_color, self.rect, self.h // 2)

        pad = 3
        knob_r = self.h // 2 - pad
        knob_x_left  = self.x + pad + knob_r
        knob_x_right = self.x + self.w - pad - knob_r
        kx = int(knob_x_left + (knob_x_right - knob_x_left) * self.knob_anim)
        ky = self.y + self.h // 2
        pygame.draw.circle(surf, WHITE, (kx, ky), knob_r)

        lbl_pen = font_sm.render(self.label_left, True,
                                 TEXT_PRI if not self.is_right else TEXT_DIM)
        lbl_era = font_sm.render(self.label_right, True,
                                 TEXT_PRI if self.is_right else TEXT_DIM)
        surf.blit(lbl_pen, (self.x - lbl_pen.get_width() - 8,
                             self.y + self.h // 2 - lbl_pen.get_height() // 2))
        surf.blit(lbl_era, (self.x + self.w + 8,
                             self.y + self.h // 2 - lbl_era.get_height() // 2))


#main

def main():
    pygame.init()
    pygame.font.init()

    info = pygame.display.Info()
    SW, SH = info.current_w, info.current_h
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((SW, SH), pygame.NOFRAME)
    pygame.display.set_caption('Akina-Draw')
    clock = pygame.time.Clock()

    def try_font(names, size):
        for n in names:
            try:
                f = pygame.font.SysFont(n, size)
                if f:
                    return f
            except Exception:
                pass
        return pygame.font.Font(None, size)

    font_title = try_font(['Bahnschrift', 'Segoe UI', 'Ubuntu', 'DejaVu Sans'], 22)
    font_body  = try_font(['Bahnschrift', 'Segoe UI', 'Ubuntu', 'DejaVu Sans'], 17)
    font_sm    = try_font(['Bahnschrift', 'Segoe UI', 'Ubuntu', 'DejaVu Sans'], 14)
    font_pred  = try_font(['Bahnschrift', 'Segoe UI', 'Ubuntu', 'DejaVu Sans'], 26)
    font_conf  = try_font(['Bahnschrift', 'Segoe UI', 'Ubuntu', 'DejaVu Sans'], 18)

    # Canvas
    CANVAS_LOGICAL = 128
    scale = max(2, min(SH // CANVAS_LOGICAL - 2, SW // CANVAS_LOGICAL - 2))
    CANVAS_DISPLAY = CANVAS_LOGICAL * scale

    cx = SW // 2 - CANVAS_DISPLAY // 2
    cy = SH // 2 - CANVAS_DISPLAY // 2

    canvas_np = np.ones((CANVAS_LOGICAL, CANVAS_LOGICAL, 3), dtype=np.uint8) * 255

    # Undo / redo history — each entry is a full canvas snapshot
    undo_stack = [canvas_np.copy()]   # current state always at the top
    redo_stack = []

    # Right panel
    RIGHT_PANEL_W = 220
    RIGHT_X = SW - RIGHT_PANEL_W - 24
    BTN_W, BTN_H = 170, 44

    bx = RIGHT_X + (RIGHT_PANEL_W - BTN_W) // 2
    btn_start = Button(bx, cy + 56,  BTN_W, BTN_H, 'START',  (50, 130, 220))
    btn_stop  = Button(bx, cy + 112, BTN_W, BTN_H, 'STOP',   (55, 58, 80))
    btn_clear = Button(bx, cy + 172, BTN_W, BTN_H, 'CLEAR',  (190, 55, 75))

    # Settings box
    SETTINGS_H = 190
    set_y = SH - SETTINGS_H - 16

    brush_slider  = Slider(RIGHT_X + 20, set_y + 62,  RIGHT_PANEL_W - 40, 1, 20, 1, 'Brush Size')
    eraser_slider = Slider(RIGHT_X + 20, set_y + 122, RIGHT_PANEL_W - 40, 1, 40, 1, 'Eraser Size')

    toggle_x = RIGHT_X + RIGHT_PANEL_W // 2 - 26
    toggle_y = set_y + 158
    tool_toggle = ToggleSwitch(toggle_x, toggle_y)

    # Left panel (Class Examples)
    LEFT_PANEL_W = 250
    CLOSED_X = -LEFT_PANEL_W + 28
    OPEN_X   = 0
    panel_anim = float(CLOSED_X)
    panel_open = False
    left_panel_target = float(CLOSED_X)

    ITEM_H = 32
    LIST_TOP = 60
    scroll_offset = 0
    list_h = SH - LIST_TOP - 10
    MAX_SCROLL = max(0, len(class_names) * ITEM_H - list_h)

    hovered_class = None

    # State
    running = False
    drawing = False
    last_lp = None

    prediction_label = 'Draw something!'
    confidence_val   = 0.0
    pred_anim        = 1.0

    # Main loop
    while True:
        dt = clock.tick(60) / 1000.0
        mouse = pygame.mouse.get_pos()

        panel_anim += (left_panel_target - panel_anim) * min(1.0, dt * 14)
        lpx = int(panel_anim)

        tool_toggle.update(dt)
        using_eraser = tool_toggle.is_right

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit()
                # E — toggle pen / eraser
                if event.key == pygame.K_e:
                    tool_toggle.is_right = not tool_toggle.is_right
                # Ctrl+Z — undo
                if event.key == pygame.K_z and (event.mod & pygame.KMOD_CTRL) and not (event.mod & pygame.KMOD_SHIFT):
                    if len(undo_stack) > 1:
                        redo_stack.append(undo_stack.pop())
                        canvas_np = undo_stack[-1].copy()
                        if running:
                            prediction_label, confidence_val = predict_canvas(canvas_np)
                            pred_anim = 0.0
                # Ctrl+Shift+Z — redo
                if event.key == pygame.K_z and (event.mod & pygame.KMOD_CTRL) and (event.mod & pygame.KMOD_SHIFT):
                    if redo_stack:
                        canvas_np = redo_stack.pop()
                        undo_stack.append(canvas_np.copy())
                        if running:
                            prediction_label, confidence_val = predict_canvas(canvas_np)
                            pred_anim = 0.0

            brush_slider.handle_event(event)
            eraser_slider.handle_event(event)
            tool_toggle.handle_event(event)

            if btn_start.handle_event(event):
                running = True
            if btn_stop.handle_event(event):
                running = False
            if btn_clear.handle_event(event):
                undo_stack.append(canvas_np.copy())
                redo_stack.clear()
                canvas_np[:] = 255
                prediction_label = 'Draw something!'
                confidence_val   = 0.0
                pred_anim = 1.0

            c_rect = pygame.Rect(cx, cy, CANVAS_DISPLAY, CANVAS_DISPLAY)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if c_rect.collidepoint(event.pos):
                    drawing = True
                    last_lp = screen_to_canvas(event.pos[0], event.pos[1], cx, cy, CANVAS_DISPLAY, CANVAS_LOGICAL)
                    # Save snapshot before stroke begins — clears redo history
                    undo_stack.append(canvas_np.copy())
                    redo_stack.clear()
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drawing = False
                last_lp = None
                if running:
                    prediction_label, confidence_val = predict_canvas(canvas_np)
                    pred_anim = 0.0

            tab_rect = pygame.Rect(lpx + LEFT_PANEL_W - 28, SH // 2 - 55, 28, 110)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if tab_rect.collidepoint(event.pos):
                    panel_open = not panel_open
                    left_panel_target = float(OPEN_X if panel_open else CLOSED_X)

            if event.type == pygame.MOUSEWHEEL:
                if pygame.Rect(lpx, LIST_TOP, LEFT_PANEL_W - 28, list_h).collidepoint(mouse):
                    scroll_offset = max(0, min(MAX_SCROLL, scroll_offset - event.y * 28))

        # Drawing drag
        if drawing:
            c_rect = pygame.Rect(cx, cy, CANVAS_DISPLAY, CANVAS_DISPLAY)
            if c_rect.collidepoint(mouse):
                lp = screen_to_canvas(mouse[0], mouse[1], cx, cy, CANVAS_DISPLAY, CANVAS_LOGICAL)
                lp = (max(0, min(CANVAS_LOGICAL - 1, lp[0])), max(0, min(CANVAS_LOGICAL - 1, lp[1])))

                if using_eraser:
                    er = max(1, int(eraser_slider.value * CANVAS_LOGICAL / CANVAS_DISPLAY))
                    if last_lp:
                        cv2.line(canvas_np, last_lp, lp, (255, 255, 255), er)
                    else:
                        cv2.circle(canvas_np, lp, er, (255, 255, 255), -1)
                else:
                    br = max(1, int(brush_slider.value * CANVAS_LOGICAL / CANVAS_DISPLAY))
                    if last_lp:
                        cv2.line(canvas_np, last_lp, lp, (0, 0, 0), br)
                    else:
                        cv2.circle(canvas_np, lp, br, (0, 0, 0), -1)

                last_lp = lp
                if running:
                    prediction_label, confidence_val = predict_canvas(canvas_np)
                    pred_anim = 0.0

        btn_start.update(mouse)
        btn_stop.update(mouse)
        btn_clear.update(mouse)
        pred_anim = min(1.0, pred_anim + dt * 3)

        # Class hover detection
        hovered_class = None
        for i, name in enumerate(class_names):
            iy = LIST_TOP + i * ITEM_H - int(scroll_offset)
            if iy + ITEM_H < LIST_TOP or iy > SH:
                continue
            ir = pygame.Rect(lpx + 8, iy, LEFT_PANEL_W - 36, ITEM_H - 4)
            if ir.collidepoint(mouse):
                hovered_class = name
                break

        # RENDER
        screen.fill(BG)

        # subtle dot grid
        for gx in range(0, SW, 36):
            for gy in range(0, SH, 36):
                pygame.draw.circle(screen, (20, 22, 32), (gx, gy), 1)

        # canvas shadow
        for i in range(12, 0, -1):
            a = int(80 * (1 - i / 12))
            s = pygame.Surface((CANVAS_DISPLAY + i * 2, CANVAS_DISPLAY + i * 2), pygame.SRCALPHA)
            pygame.draw.rect(s, (0, 0, 0, a), (0, 0, CANVAS_DISPLAY + i * 2, CANVAS_DISPLAY + i * 2), border_radius=4)
            screen.blit(s, (cx - i, cy - i))

        # canvas
        canvas_surf = pygame.surfarray.make_surface(canvas_np.swapaxes(0, 1))
        canvas_disp = pygame.transform.scale(canvas_surf, (CANVAS_DISPLAY, CANVAS_DISPLAY))
        screen.blit(canvas_disp, (cx, cy))
        pygame.draw.rect(screen, PANEL_BORDER, (cx - 1, cy - 1, CANVAS_DISPLAY + 2, CANVAS_DISPLAY + 2), 2, border_radius=3)

        # canvas label
        clbl = font_sm.render('128 x 128 px canvas', True, TEXT_DIM)
        screen.blit(clbl, (cx + CANVAS_DISPLAY // 2 - clbl.get_width() // 2, cy + CANVAS_DISPLAY + 10))

        # Prediction strip
        py_base = cy + CANVAS_DISPLAY + 34
        pa = int(pred_anim * 255)

        if confidence_val > 0.5:
            pc = GREEN
        elif confidence_val > 0.25:
            pc = ORANGE
        else:
            pc = TEXT_SEC

        ps = font_pred.render(prediction_label, True, pc)
        ps.set_alpha(pa)
        screen.blit(ps, (SW // 2 - ps.get_width() // 2, py_base))

        if confidence_val > 0:
            cs = font_conf.render(f'{confidence_val * 100:.1f}% confidence', True, TEXT_SEC)
            cs.set_alpha(pa)
            screen.blit(cs, (SW // 2 - cs.get_width() // 2, py_base + 34))

            bar_w = 180
            bx2 = SW // 2 - bar_w // 2
            by2 = py_base + 62
            rounded_rect(screen, (28, 30, 45), (bx2, by2, bar_w, 5), 3)
            fw = int(confidence_val * bar_w)
            if fw > 0:
                rounded_rect(screen, pc, (bx2, by2, fw, 5), 3)

        # Right panel
        rp_s = pygame.Surface((RIGHT_PANEL_W, CANVAS_DISPLAY + 60), pygame.SRCALPHA)
        rp_s.fill((20, 22, 32, 210))
        pygame.draw.rect(rp_s, PANEL_BORDER + (255,), (0, 0, RIGHT_PANEL_W, CANVAS_DISPLAY + 60), 1, border_radius=14)
        screen.blit(rp_s, (RIGHT_X, cy - 22))

        rh = font_title.render('Controls', True, TEXT_PRI)
        screen.blit(rh, (RIGHT_X + RIGHT_PANEL_W // 2 - rh.get_width() // 2, cy - 6))

        sc = font_sm.render('* PREDICTING' if running else '* IDLE', True, GREEN if running else TEXT_DIM)
        screen.blit(sc, (RIGHT_X + RIGHT_PANEL_W // 2 - sc.get_width() // 2, cy + 28))

        btn_start.draw(screen, font_body)
        btn_stop.draw(screen, font_body)
        btn_clear.draw(screen, font_body)

        # Settings panel
        ss = pygame.Surface((RIGHT_PANEL_W, SETTINGS_H), pygame.SRCALPHA)
        ss.fill((20, 22, 32, 210))
        pygame.draw.rect(ss, PANEL_BORDER + (255,), (0, 0, RIGHT_PANEL_W, SETTINGS_H), 1, border_radius=14)
        screen.blit(ss, (RIGHT_X, set_y))

        sh_t = font_title.render('Settings', True, TEXT_PRI)
        screen.blit(sh_t, (RIGHT_X + RIGHT_PANEL_W // 2 - sh_t.get_width() // 2, set_y + 12))

        brush_slider.draw(screen, font_sm)
        eraser_slider.draw(screen, font_sm)
        tool_toggle.draw(screen, font_sm)

        # Left panel
        lp_s = pygame.Surface((LEFT_PANEL_W, SH), pygame.SRCALPHA)
        lp_s.fill((18, 20, 30, 235))
        pygame.draw.rect(lp_s, PANEL_BORDER + (255,), (0, 0, LEFT_PANEL_W, SH), 1)
        screen.blit(lp_s, (lpx, 0))

        pt = font_title.render('Class Examples', True, TEXT_PRI)
        screen.blit(pt, (lpx + (LEFT_PANEL_W - 28) // 2 - pt.get_width() // 2, 18))
        pygame.draw.line(screen, PANEL_BORDER, (lpx + 8, 46), (lpx + LEFT_PANEL_W - 36, 46), 1)

        # clipped list
        list_clip = pygame.Rect(lpx + 8, LIST_TOP, LEFT_PANEL_W - 36, list_h)
        old_clip = screen.get_clip()
        screen.set_clip(list_clip)

        for i, name in enumerate(class_names):
            iy = LIST_TOP + i * ITEM_H - int(scroll_offset)
            if iy + ITEM_H < LIST_TOP or iy > SH:
                continue
            is_hov = (name == hovered_class)
            ir = pygame.Rect(lpx + 8, iy, LEFT_PANEL_W - 36, ITEM_H - 4)
            if is_hov:
                rounded_rect(screen, (35, 55, 90), ir, 6)
            nt = font_sm.render(name, True, TEXT_PRI if is_hov else TEXT_SEC)
            screen.blit(nt, (lpx + 16, iy + (ITEM_H - 4 - nt.get_height()) // 2))

        screen.set_clip(old_clip)

        # Scrollbar
        if MAX_SCROLL > 0:
            sb_h = max(20, int(list_h / (len(class_names) * ITEM_H) * list_h))
            sb_y = LIST_TOP + int(scroll_offset / MAX_SCROLL * (list_h - sb_h))
            pygame.draw.rect(screen, ACCENT, (lpx + LEFT_PANEL_W - 9, sb_y, 4, sb_h), border_radius=2)

        # Toggle tab
        tab_r = pygame.Rect(lpx + LEFT_PANEL_W - 28, SH // 2 - 55, 28, 110)
        rounded_rect(screen, (28, 30, 44), tab_r, 8)
        pygame.draw.rect(screen, PANEL_BORDER, tab_r, 1, border_radius=8)
        arr = font_sm.render('<' if panel_open else '>', True, ACCENT)
        screen.blit(arr, (tab_r.x + tab_r.w // 2 - arr.get_width() // 2,
                          tab_r.y + tab_r.h // 2 - arr.get_height() // 2))

        # Hover preview
        if hovered_class:
            pv = load_example(hovered_class)
            px2 = lpx + LEFT_PANEL_W + 12
            py2 = max(10, min(mouse[1] - EXAMPLE_SIZE // 2, SH - EXAMPLE_SIZE - 44))
            px2 = max(0, min(SW - EXAMPLE_SIZE - 16, px2))

            bg2 = pygame.Surface((EXAMPLE_SIZE + 16, EXAMPLE_SIZE + 40), pygame.SRCALPHA)
            bg2.fill((22, 24, 36, 248))
            pygame.draw.rect(bg2, PANEL_BORDER + (255,), (0, 0, EXAMPLE_SIZE + 16, EXAMPLE_SIZE + 40), 1, border_radius=10)
            screen.blit(bg2, (px2 - 8, py2 - 8))
            screen.blit(pv, (px2, py2))
            cn_t = font_sm.render(hovered_class, True, TEXT_PRI)
            screen.blit(cn_t, (px2 + EXAMPLE_SIZE // 2 - cn_t.get_width() // 2, py2 + EXAMPLE_SIZE + 6))

        # Top bar
        tb = pygame.Surface((SW, 38), pygame.SRCALPHA)
        tb.fill((13, 13, 20, 220))
        screen.blit(tb, (0, 0))
        tt = font_title.render('Akina-Draw', True, TEXT_PRI)
        screen.blit(tt, (SW // 2 - tt.get_width() // 2, 9))
        ht = font_sm.render('ESC / Q  quit   |   E  toggle eraser', True, TEXT_DIM)
        screen.blit(ht, (SW - ht.get_width() - 14, 12))

        pygame.display.flip()

    pygame.quit()


def screen_to_canvas(sx, sy, cx, cy, disp, logical):
    lx = (sx - cx) / disp * logical
    ly = (sy - cy) / disp * logical
    return int(lx), int(ly)


if __name__ == '__main__':
    main()