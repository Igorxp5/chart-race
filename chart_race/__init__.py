import io
import os
import cv2
import tqdm
import math
import random
import typing
import locale
import logging
import colorsys
import datetime
import calendar
import dataclasses
import numpy as np

from .models import Entity, Entry

from PIL import Image, ImageDraw, ImageFont

# Paths
ROOT_DIR = os.path.dirname(__file__)
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
FONTS_DIR = os.path.join(RESOURCES_DIR, 'fonts')
IMAGES_DIR = os.path.join(RESOURCES_DIR, 'images')

# Style Constants
COLOR_WHITE = 255, 255, 255, 255
COLOR_WHITE_1 = 248, 249, 250, 255
COLOR_LIGHT_GRAY = 229, 229, 230, 255
COLOR_DARK_GRAY = 128, 128, 128, 255
COLOR_DARK_GRAY_1 = 80, 80, 80, 255
COLOR_DARK_GRAY_2 = 21, 21, 21, 255

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_MODE = 'RGBA'
IMAGE_SIZE = IMAGE_WIDTH, IMAGE_HEIGHT
IMAGE_BACKGROUND = COLOR_WHITE_1
DEFAULT_LANGUAGE = 'pt'

RESOLUTION_FACTOR  = 1920 / 1920

HORIZONTAL_PADDING = math.ceil(80 * RESOLUTION_FACTOR)
VERTICAL_PADDING = math.ceil(30 * RESOLUTION_FACTOR)

# Fonts Constants
ROBOTO_FONT_PATH = os.path.join(FONTS_DIR, 'Roboto-Regular.ttf')
ROBOTO_BOLD_FONT_PATH = os.path.join(FONTS_DIR, 'Roboto-Bold.ttf')
ROBOTO_LIGHT_FONT_PATH = os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
PT_TO_PX = 1.3281472327
PT_TO_PX_RATE = 0.9

# Title text
TITLE_DEFAULT_TEXT = 'CHART RACE'
TITLE_COLOR = COLOR_DARK_GRAY_1
TITLE_TEXT_SIZE = math.ceil(70 * RESOLUTION_FACTOR)
TITLE_TEXT_FONT = ImageFont.truetype(ROBOTO_BOLD_FONT_PATH, TITLE_TEXT_SIZE)
TITLE_BASE_Y = VERTICAL_PADDING

# Axis constants
AXIS_COLOR = COLOR_DARK_GRAY_1
AXIS_STROKE_WIDTH = 4

# Chart bar constants
CHART_BAR_TOTAL_ENTITIES = 10
CHART_BAR_HEIGHT = math.ceil(58 * RESOLUTION_FACTOR)
CHART_BAR_VERTICAL_MARGIN = math.ceil(28 * RESOLUTION_FACTOR)
CHART_BAR_LABEL_COLOR = COLOR_DARK_GRAY_2
CHART_BAR_LABEL_HEIGHT = math.ceil(50 * RESOLUTION_FACTOR)
CHART_BAR_LABEL_MARGIN = math.ceil(20 * RESOLUTION_FACTOR)
CHART_BAR_LABEL_VERTICAL_PADDING = math.ceil(10 * RESOLUTION_FACTOR)
CHART_BAR_LABEL_HORIZONTAL_PADDING = math.ceil(20 * RESOLUTION_FACTOR)
CHART_BAR_LABEL_TEXT_SIZE = math.ceil(25 * RESOLUTION_FACTOR)
CHART_BAR_LABEL_TEXT_FONT = ImageFont.truetype(ROBOTO_LIGHT_FONT_PATH, CHART_BAR_LABEL_TEXT_SIZE)
CHART_BAR_LABEL_TEXT_COLOR = COLOR_WHITE
CHART_BAR_VALUE_TEXT_SIZE = math.ceil(26 * RESOLUTION_FACTOR)
CHART_BAR_VALUE_TEXT_FONT = ImageFont.truetype(ROBOTO_FONT_PATH, CHART_BAR_VALUE_TEXT_SIZE)
CHART_BAR_VALUE_TEXT_COLOR = COLOR_DARK_GRAY_2
CHART_BAR_VALUE_MARGIN = math.ceil(10 * RESOLUTION_FACTOR)
CHART_BAR_HEIGHT_AND_MARGIN = CHART_BAR_HEIGHT + CHART_BAR_VERTICAL_MARGIN
CHART_BAR_DISABLED_OPACITY = 0.15

# Random color range
CHART_BAR_COLOR_HUE_RANGE = [0.1, 0.9]
CHART_BAR_COLOR_SATURATION_RANGE = [0.4, 0.55]
CHART_BAR_COLOR_BRIGHTNESS_RANGE = [0.2, 0.5]
MAX_ENTITY_BAR_COLOR_BRIGHTNESS = 0.6
MIN_ENTITY_BAR_COLOR_SATURATION = 0.6

# Chart constants
CHART_WIDTH = math.ceil(1200 * RESOLUTION_FACTOR)
CHART_HEIGHT = CHART_BAR_TOTAL_ENTITIES * CHART_BAR_HEIGHT + (CHART_BAR_TOTAL_ENTITIES - 1) * CHART_BAR_VERTICAL_MARGIN

# Grid constants
GRID_TOTAL_COLUMNS = 8
GRID_COLOR = COLOR_LIGHT_GRAY
GRID_STROKE_WIDTH = 1
GRID_SCALE_VERTICAL_MARGIN = math.ceil(20 * RESOLUTION_FACTOR)
GRID_SCALE_TEXT_SIZE = math.ceil(25 * RESOLUTION_FACTOR)
GRID_SCALE_TEXT_COLOR = COLOR_DARK_GRAY_1
GRID_SCALE_TEXT_FONT = ImageFont.truetype(ROBOTO_FONT_PATH, GRID_SCALE_TEXT_SIZE)
GRID_SCALE_TEXT_MARGIN = math.ceil(10 * RESOLUTION_FACTOR)
GRID_SCALE_BASE_Y = TITLE_BASE_Y + TITLE_TEXT_SIZE + GRID_SCALE_VERTICAL_MARGIN

GRID_MIN_SCALE = 250
GRID_FIRST_ENTITY_SCALE_FACTOR = 1.10
GRID_SCALE_ANIMATION = [[2, 1], [5, 2], [10, 4], [20, 10]] # [Divisors, Limits to the divisors] 

# Chart position
CHART_BASE_X = (IMAGE_WIDTH - CHART_WIDTH) // 2 + AXIS_STROKE_WIDTH
CHART_BASE_Y = GRID_SCALE_BASE_Y + GRID_SCALE_TEXT_SIZE + GRID_SCALE_VERTICAL_MARGIN

# Date text
DATE_TEXT_SIZE = 60
DATE_FONT = ImageFont.truetype(ROBOTO_BOLD_FONT_PATH, DATE_TEXT_SIZE)
DATE_COLOR = COLOR_DARK_GRAY

# Video settings
VIDEO_FRAME_RATE = 60
VIDEO_END_FREEZE_TIME = 5
DEFAULT_DAYS_PER_SECOND = 7
DEFAULT_VIDEO_SPEED = 1.5  # x1.5 (drop frames)
DEFAULT_ELAPSED_TIMESTAMP_BY_FRAME = (86400 * DEFAULT_DAYS_PER_SECOND) / VIDEO_FRAME_RATE

# Animation settings
SCALE_GROWTH_RATE = 1
ANIMATION_SMOOTHNESS = 1  # The higher the smoothness, the less accurate the amount of entries on the time scale

# Date transition animation settings
DATE_TRANSITION_DURATION = 0.7
DATE_TRANSITION_TOTAL_FRAMES = math.ceil(DATE_TRANSITION_DURATION * VIDEO_FRAME_RATE)
DATE_TRANSITION_Y_OFFSET = 50
DATE_TRANSITION_Y_OFFSET_STEP = DATE_TRANSITION_Y_OFFSET / (DATE_TRANSITION_DURATION * VIDEO_FRAME_RATE)
DATE_TRANSITION_OPACITY_STEP = 1 / (DATE_TRANSITION_DURATION * VIDEO_FRAME_RATE)

# Chart bar transition animation settings
CHART_BAR_TRANSITION_DURATION = 0.4 
CHART_BAR_TRANSITION_TOTAL_FRAMES = math.ceil(CHART_BAR_TRANSITION_DURATION * VIDEO_FRAME_RATE)
CHART_BAR_FADE_IN_Y = CHART_BASE_Y + ((CHART_BAR_TOTAL_ENTITIES - 1) * CHART_BAR_HEIGHT_AND_MARGIN) + 1
CHART_BAR_FADE_OUT_MAX_Y = CHART_BASE_Y + CHART_HEIGHT + CHART_BAR_HEIGHT_AND_MARGIN

DEFAULT_ENTITY_IMAGE = Image.open(os.path.join(IMAGES_DIR, 'default-image.png'))
DATE_FORMAT = '%b %Y'

RgbColor = typing.Tuple[int, int, int]
RgbaColor = typing.Tuple[int, int, int, int]
ColorType = RgbColor | RgbaColor | str
Coordinate = typing.Tuple[int, int]

@dataclasses.dataclass
class EntityBar:
    color: RgbColor
    entity_label: str = None
    value: float = 0
    entity_image: Image.Image = None
    enabled: bool = True


@dataclasses.dataclass
class PodiumEntity:
    entity: Entity
    value: float
    max_value: float = float('inf')

    def __hash__(self):
        return hash(f'_____{self.__class__.__name__}_____{self.entity.key}_____')


@dataclasses.dataclass
class EntityBarAnimationState:
    podium_index: int
    entity_bar: EntityBar
    position: Coordinate = (0, 0)
    final_position: Coordinate = (0, 0)
    layer: int = 0
    opacity: float = 1
    position_transition_left_frames: int = 0 


@dataclasses.dataclass
class AnimationState:
    current_frame: int = 0
    scale: float = GRID_MIN_SCALE
    base_image: Image.Image = None
    current_date: datetime.datetime = None
    group_entry_range_timedelta: datetime.timedelta = None
    frame_step_timedelta: datetime.timedelta = None
    date_transition_start_frame: int = None
    entity_bar_states: typing.Dict[PodiumEntity, EntityBarAnimationState] = dataclasses.field(default_factory=lambda: {})


class Podium:
    def __init__(self, entities: typing.Iterable[Entity], entries: typing.List[Entry]):
        self._podium_entities: typing.Dict[str, PodiumEntity] = dict()
        self._podium: typing.List[PodiumEntity] = []
        self._sorted = False
        for entity in entities:
            entity_entries = [e for e in entries if e.entity_key == entity.key]
            podium_entity = PodiumEntity(entity, 0, len(entity_entries))
            self._podium.append(podium_entity)
            self._podium_entities[entity.key] = podium_entity

    def __len__(self):
        return len(self._podium)
    
    def __iter__(self) -> typing.Iterator[PodiumEntity]:
        if not self._sorted:
            self._podium.sort(reverse=True, key=lambda entity: entity.value)
            self._sorted = True
        return iter(self._podium)

    def __getitem__(self, index) -> PodiumEntity:
        if not self._sorted:
            self._podium.sort(reverse=True, key=lambda entity: entity.value)
            self._sorted = True
        return self._podium[index]

    def increment_entity_entries(self, key, increment):
        max_increment = self._podium_entities[key].max_value - self._podium_entities[key].value
        if self._podium_entities[key].value < self._podium_entities[key].max_value:
            self._podium_entities[key].value += min(increment, max_increment)
            self._sorted = False
        else:
            logging.warning(f'Cannot increment {increment} to the key: {key}. Reached max value.')


def get_offset_center(image_size: typing.Tuple[int, int], obj_size: typing.Tuple[int, int]):
    return (image_size[0] - obj_size[0]) // 2, (image_size[1] - obj_size[1]) // 2


def get_text_size(text: str, font: ImageFont.FreeTypeFont, letter_spacing=0) -> typing.Tuple[int, int]:
    width, height = font.getsize(text)
    width += letter_spacing * (len(text) - 1) * 0.75
    return math.ceil(width), math.ceil(height)


def center_text(image: Image.Image, text: str, font: ImageFont.FreeTypeFont, fill: ColorType,
                x: int, y: int, width: int=None, height: int=None):
    text_size = font.getsize(text)
    if width:
        x += (width - text_size[0]) // 2
    if height:
        y += (height - text_size[1]) // 2
    draw_text(image, text, fill, (x, y), font)


def round_corner(radius: int, fill: ColorType):
    corner = Image.new(IMAGE_MODE, (radius, radius), (0, 0, 0, 0))
    draw = ImageDraw.Draw(corner)
    draw.pieslice((0, 0, radius * 2, radius * 2), 180, 270, fill=fill)
    return corner


def round_rectangle(size: typing.Tuple[int, int], radius: int, fill: ColorType) -> Image.Image:
    width, height = size
    rectangle = Image.new(IMAGE_MODE, size, fill)
    corner = round_corner(radius, fill)
    rectangle.paste(corner, (0, 0))
    rectangle.paste(corner.rotate(90), (0, height - radius)) # Rotate the corner and paste it
    rectangle.paste(corner.rotate(180), (width - radius, height - radius))
    rectangle.paste(corner.rotate(270), (width - radius, 0))
    return rectangle


def mask_image_by_circle(image: Image.Image, opacity: float=1) -> Image.Image:
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    fill = max(0, min(255, int(opacity * 255)))
    draw.ellipse((0, 0) + image.size, fill=fill)
    mask = mask.resize(image.size, Image.ANTIALIAS)
    image.putalpha(mask)
    return image


def create_entity_image(entity_image: Image.Image, size: typing.Tuple[int, int], opacity: float=1) -> Image.Image:
    entity_image = mask_image_by_circle(entity_image, opacity)
    entity_image = entity_image.resize(size, Image.ANTIALIAS)
    return entity_image


def scale_entity_bar_width(value: float, scale: float) -> int:
    return max(int((CHART_WIDTH / scale) * value), CHART_BAR_HEIGHT)


def color_opacity(color: typing.Union[RgbColor, RgbaColor], opacity: float) -> RgbaColor:
    opacity = max(0, min(255, int(opacity * 255)))
    return color[:3] + (opacity,)


def choose_entity_bar_color(entity: Entity) -> RgbaColor:
    if not entity.image_path:
        return random_color(CHART_BAR_COLOR_HUE_RANGE, CHART_BAR_COLOR_SATURATION_RANGE, CHART_BAR_COLOR_BRIGHTNESS_RANGE)
    image = Image.open(entity.image_path)
    total_pixels = image.size[0] * image.size[1]
    dominant_color = max([c for c in image.getcolors(total_pixels) if len(c[1]) >= 3 or c[1][3] >= 70], key=lambda c: c[0])[1]
    dominant_color = [c / 255 for c in dominant_color]
    dominant_color = [dominant_color[i] for i in range(3)]  # remove alpha if it's present
    hue, saturation, brightness = colorsys.rgb_to_hsv(*dominant_color)
    r, g, b = colorsys.hsv_to_rgb(hue, max(saturation, MIN_ENTITY_BAR_COLOR_SATURATION), min(brightness, MAX_ENTITY_BAR_COLOR_BRIGHTNESS))
    return int(r * 256), int(g * 256), int(b * 256), 255


def random_color(hue_range: typing.Iterable[int], saturation_range: typing.Iterable[int],
                 brightness_range: typing.Iterable[int]) -> RgbaColor:
    hue = random.uniform(*hue_range)
    saturation = random.uniform(*saturation_range)
    brightness_range = random.uniform(*brightness_range)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness_range)
    return int(r * 256), int(g * 256), int(b * 256), 255


def draw_text(image: Image.Image, text: str, fill: ColorType, position: Coordinate,
              font: ImageFont.FreeTypeFont, letter_spacing: int=0):
    width, height = get_text_size(text, font, letter_spacing)

    text_placeholder = Image.new('RGBA', (width + 2, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_placeholder)

    x, y = position
    text_center_offset = get_offset_center(image.size, (width, height))
    x = text_center_offset[0] if x == 'center' else x
    y = text_center_offset[1] // 2 if y == 'center' else y

    x, y = int(x), int(y)

    drawed_width = 0
    for _, letter in enumerate(text):
        draw.text((drawed_width, 0), letter, fill=fill, font=font)
        drawed_width += font.getsize(letter)[0] + letter_spacing

    image.paste(text_placeholder, (x, y), mask=text_placeholder)


def draw_title(image: Image.Image, title: str):
    center_text(image, title, TITLE_TEXT_FONT, TITLE_COLOR, 
                0, TITLE_BASE_Y, width=IMAGE_WIDTH)


def draw_date(image: Image.Image, date: datetime.datetime, y_offset: int=0, opacity: float=1):
    opacity = int(max(0, min(255, opacity * 255)))
    color = DATE_COLOR[:3] + (opacity,)
    text = date.strftime(DATE_FORMAT).upper()
    width, height = get_text_size(text, DATE_FONT)
    x = IMAGE_WIDTH - width - HORIZONTAL_PADDING
    y = IMAGE_HEIGHT - VERTICAL_PADDING - height + y_offset
    draw_text(image, text, color, (x, y), DATE_FONT)


def draw_scale(image: Image.Image, scale: float):
    scale = int(scale)

    scale_ten_power = 10 ** (len(str(scale)) - 1)
    divisor = 0

    for base_divisor, base_limit in GRID_SCALE_ANIMATION:
        if scale <= base_limit * scale_ten_power:
            divisor = base_divisor * scale_ten_power / 10
            break

    total_markers = math.floor(scale / divisor)

    draw = ImageDraw.Draw(image)

    grid_unit_width = CHART_WIDTH / (scale / divisor)
    for c in range(total_markers + 1):
        x = CHART_BASE_X + grid_unit_width * c
        value = int(c * divisor)
        text = '{:,}'.format(value)
        text_width, text_height = get_text_size(text, GRID_SCALE_TEXT_FONT)
        draw_text(image, text, GRID_SCALE_TEXT_COLOR, (x - text_width // 2, GRID_SCALE_BASE_Y), GRID_SCALE_TEXT_FONT)
        x1 = x2 = x
        y1 = GRID_SCALE_BASE_Y + text_height + GRID_SCALE_TEXT_MARGIN 
        y2 = CHART_BASE_Y + CHART_HEIGHT
        draw.line([x1, y1, x2, y2], GRID_COLOR, GRID_STROKE_WIDTH)


def draw_entity_bar(image: Image.Image, x: int, y: int, width: int,
                    entity_image: Image.Image, entity_label: str, value: float,
                    color: typing.Union[RgbColor, RgbaColor], opacity: float):
    x -= CHART_BAR_LABEL_MARGIN

    # Label
    text_width, text_height = get_text_size(entity_label, CHART_BAR_LABEL_TEXT_FONT)

    label_width = text_width + CHART_BAR_LABEL_HORIZONTAL_PADDING * 2
    label_x = x - label_width
    label_y = y + (CHART_BAR_HEIGHT - CHART_BAR_LABEL_HEIGHT) // 2

    text_box = round_rectangle((label_width, CHART_BAR_LABEL_HEIGHT), 0, color_opacity(CHART_BAR_LABEL_COLOR, opacity))
    image.paste(text_box, (label_x, label_y), text_box.convert(IMAGE_MODE))
    
    center_text(image, entity_label, CHART_BAR_LABEL_TEXT_FONT, color_opacity(CHART_BAR_LABEL_TEXT_COLOR, opacity),
                label_x, label_y, label_width, CHART_BAR_LABEL_HEIGHT)

    x += CHART_BAR_LABEL_MARGIN

    # Bar
    radius = CHART_BAR_HEIGHT // 2
    bar = round_rectangle((width, CHART_BAR_HEIGHT), radius, color_opacity(color, opacity))
    image.paste(bar, (x, y), bar.convert(IMAGE_MODE))

    # Value
    text_x = x + width + CHART_BAR_VALUE_MARGIN
    text_y = y + (CHART_BAR_HEIGHT - text_height) / 2
    draw_text(image, '{:,}'.format(value), color_opacity(CHART_BAR_VALUE_TEXT_COLOR, opacity), 
              (text_x, text_y), CHART_BAR_VALUE_TEXT_FONT)
    
    if opacity < 1:
        entity_image = create_entity_image(entity_image.copy(), (CHART_BAR_HEIGHT, CHART_BAR_HEIGHT), opacity)

    image.paste(entity_image, (x + width - CHART_BAR_HEIGHT, y), entity_image.convert(IMAGE_MODE))


def frame_generate_base_image(title: str) -> Image.Image:
    image = Image.new(IMAGE_MODE, IMAGE_SIZE, color=IMAGE_BACKGROUND)
    draw_title(image, title)
    return image


def frame_handle_date_transition(image: Image.Image, animation_state: AnimationState):
    current_month_last_day = calendar.monthrange(animation_state.current_date.year, animation_state.current_date.month)[1]
    next_month = datetime.datetime(year=animation_state.current_date.year, month=animation_state.current_date.month, day=current_month_last_day)
    next_month += datetime.timedelta(days=1)
    remaining_frames_to_next_month = math.ceil((next_month - animation_state.current_date) / animation_state.frame_step_timedelta)
    is_month_close_to_change = remaining_frames_to_next_month <= DATE_TRANSITION_TOTAL_FRAMES
    
    if not animation_state.date_transition_start_frame and is_month_close_to_change:
        animation_state.date_transition_start_frame = animation_state.current_frame
    
    if animation_state.date_transition_start_frame and not is_month_close_to_change:
        animation_state.date_transition_start_frame = None

    if not animation_state.date_transition_start_frame:
        draw_date(image, animation_state.current_date)
    else:
        date_transition_frame = animation_state.current_frame - animation_state.date_transition_start_frame

        current_date_y_offset = int(-DATE_TRANSITION_Y_OFFSET_STEP * date_transition_frame)
        current_date_opacity = 1 - (DATE_TRANSITION_OPACITY_STEP * date_transition_frame)
        draw_date(image, animation_state.current_date, y_offset=current_date_y_offset, opacity=current_date_opacity)

        next_date_y_offset = DATE_TRANSITION_Y_OFFSET - (DATE_TRANSITION_Y_OFFSET_STEP * date_transition_frame)
        next_date_opacity = DATE_TRANSITION_OPACITY_STEP * date_transition_frame
        draw_date(image, next_month, y_offset=next_date_y_offset, opacity=next_date_opacity)


def frame(title: str, animation_state: AnimationState) -> Image.Image:
    if not animation_state.base_image:
        animation_state.base_image = image = frame_generate_base_image(title)

    image = animation_state.base_image.copy()

    frame_handle_date_transition(image, animation_state)

    draw_scale(image, animation_state.scale)

    # Draw low layers first
    entity_bar_states = sorted(animation_state.entity_bar_states.values(), key=lambda s: s.layer)
    for entity_bar_state in entity_bar_states:
        if entity_bar_state.position[1] < CHART_BAR_FADE_OUT_MAX_Y:
            entity_bar = entity_bar_state.entity_bar
            bar_width = int(scale_entity_bar_width(entity_bar.value, animation_state.scale))
            draw_entity_bar(image, *entity_bar_state.position, bar_width, entity_bar.entity_image, entity_bar.entity_label, 
                            entity_bar.value, entity_bar.color, entity_bar_state.opacity)

    return image


def generate_frame_data(animation_state: AnimationState, podium: Podium,
                        entity_images: typing.Dict[str, Image.Image],
                        entity_colors: typing.Dict[str, RgbColor],
                        gray_out_disabled_entities: bool):
    if not animation_state.entity_bar_states:
        animation_state.entity_bar_states = dict()
        for i, podium_entity in enumerate(podium):
            label = podium_entity.entity.label
            entity_image = entity_images[podium_entity.entity.key]
            entity_color = entity_colors[podium_entity.entity.key]
            entity_bar = EntityBar(entity_color, label, entity_image=entity_image)
            animation_state.entity_bar_states[podium_entity] = entity_bar_state = EntityBarAnimationState(i, entity_bar)
            entity_bar_state.podium_index = -1
            entity_bar_state.position = CHART_BASE_X, CHART_BASE_Y + ((CHART_BAR_TOTAL_ENTITIES + 1) * CHART_BAR_HEIGHT_AND_MARGIN)
            entity_bar_state.final_position = entity_bar_state.position

    for i, podium_entity in enumerate(podium):
        entity_bar_state = animation_state.entity_bar_states[podium_entity]
        entity_bar_state.entity_bar.value = int(podium_entity.value)
        if gray_out_disabled_entities and podium_entity.value >= podium_entity.max_value:
            entity_bar_state.entity_bar.enabled = False
        entity_bar_state.layer = len(podium) - i

        if i != entity_bar_state.podium_index and entity_bar_state.entity_bar.value > 0:
            entity_bar_state.podium_index = i
            entity_bar_state.position_transition_left_frames = CHART_BAR_TRANSITION_TOTAL_FRAMES
            entity_bar_state.final_position = CHART_BASE_X, CHART_BASE_Y + i * CHART_BAR_HEIGHT_AND_MARGIN

        if entity_bar_state.position_transition_left_frames > 0:
            x_step = (entity_bar_state.final_position[0] - entity_bar_state.position[0]) // entity_bar_state.position_transition_left_frames
            y_step = (entity_bar_state.final_position[1] - entity_bar_state.position[1]) // entity_bar_state.position_transition_left_frames
            entity_bar_state.position = entity_bar_state.position[0] + x_step, entity_bar_state.position[1] + y_step

            entity_bar_state.position_transition_left_frames -= 1

        base_opacity = 1
        if not entity_bar_state.entity_bar.enabled:
            base_opacity = CHART_BAR_DISABLED_OPACITY

        if entity_bar_state.position[1] < CHART_BAR_FADE_IN_Y:
            entity_bar_state.opacity = base_opacity
        else:
            entity_bar_state.opacity = entity_bar_state.position[1] - CHART_BAR_FADE_IN_Y
            entity_bar_state.opacity = max(0, min(base_opacity, 1 - (entity_bar_state.opacity / CHART_BAR_HEIGHT)))

    first_entity_based_scale = podium[0].value * GRID_FIRST_ENTITY_SCALE_FACTOR
    animation_state.scale = max(GRID_MIN_SCALE, first_entity_based_scale, animation_state.scale * SCALE_GROWTH_RATE)


def generate_video_frames(entries: typing.Iterable[Entry], title: str,
                          start_date: datetime.datetime,
                          video_speed: float,
                          frame_step_timedelta: datetime.timedelta,
                          podium: Podium, entity_images: typing.Dict[str, Image.Image],
                          entity_colors: typing.Dict[str, RgbColor],
                          gray_out_disabled_entities: bool) -> typing.Generator[Image.Image, None, None]:
    frame_image: Image.Image = None
    group_entries_range_timedelta = datetime.timedelta(days=7 * ANIMATION_SMOOTHNESS)
    animation_state = AnimationState(current_date=start_date, frame_step_timedelta=frame_step_timedelta,
                                     group_entry_range_timedelta=group_entries_range_timedelta)
    next_step_date: datetime.datetime = animation_state.current_date + group_entries_range_timedelta
    frames_by_group: int = group_entries_range_timedelta // frame_step_timedelta
    entity_total_entries: typing.Dict[str, int] = dict()
    for entry in entries:
        if entry.date < next_step_date:
            entity_total_entries.setdefault(entry.entity_key, 0)
            entity_total_entries[entry.entity_key] += 1
        else:
            for key in entity_total_entries:
                entity_total_entries[key] = entity_total_entries[key] / frames_by_group

            for _ in range(frames_by_group):
                for key in entity_total_entries:
                    podium.increment_entity_entries(key, entity_total_entries[key])

                if video_speed > 0 or animation_state.current_frame % (1 / video_speed) >= 1:
                    generate_frame_data(animation_state, podium, entity_images, entity_colors,
                                        gray_out_disabled_entities)

                if video_speed <= 0 or animation_state.current_frame % video_speed < 1:
                    frame_image = frame(title, animation_state)
                    yield frame_image

                animation_state.current_frame += 1
                animation_state.current_date += frame_step_timedelta

            next_step_date = animation_state.current_date + group_entries_range_timedelta
            entity_total_entries = dict()

    # TODO: Last entries aren't being included

    if frame_image:
        for _ in range(VIDEO_END_FREEZE_TIME * VIDEO_FRAME_RATE):
            yield frame_image


def create_chart_race_video(entities: typing.List[Entity], entries: typing.List[Entry],
                            output: str, title: str = TITLE_DEFAULT_TEXT,
                            gray_out_disabled_entities: bool = False, locale_='en_US',
                            video_speed=DEFAULT_VIDEO_SPEED,
                            elapsed_timestamp_by_frame=DEFAULT_ELAPSED_TIMESTAMP_BY_FRAME):
    logging.info('Sorting entries by date...')
    entries.sort(key=lambda entry: entry.date)
    logging.info('Entries sorted!')

    logging.info('Rendering entity images...')
    entity_images = dict()
    entity_colors = dict()
    for entity in entities:
        if entity.image_path:
            entity_images[entity.key] = Image.open(entity.image_path)
        else:
            entity_images[entity.key] = DEFAULT_ENTITY_IMAGE
        entity_images[entity.key] = create_entity_image(entity_images[entity.key], (CHART_BAR_HEIGHT, CHART_BAR_HEIGHT))
        entity_colors[entity.key] = choose_entity_bar_color(entity)

    start_date = entries[0].date
    end_date = entries[-1].date

    podium = Podium(entities, entries)
    
    logging.info('Rendering video...')
    total_frames = ((end_date - start_date).total_seconds() / elapsed_timestamp_by_frame) // video_speed
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output, fourcc, VIDEO_FRAME_RATE, IMAGE_SIZE)
    frame_step_timedelta = datetime.timedelta(seconds=elapsed_timestamp_by_frame)

    default_locale = f'{locale.getdefaultlocale()[0]}.UTF-8'
    locale.setlocale(locale.LC_ALL, f'{locale_}.UTF-8')

    try:
        frames = generate_video_frames(entries, title, start_date, video_speed, frame_step_timedelta, podium,
                                       entity_images, entity_colors, gray_out_disabled_entities)
        tqdm_iterator = tqdm.tqdm(frames, total=total_frames)
        try:
            for frame in tqdm_iterator:
                cv_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGR)
                video_writer.write(cv_image)
        finally:
            video_writer.release()
    finally:
        locale.setlocale(locale.LC_ALL, default_locale)
