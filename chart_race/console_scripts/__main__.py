import csv
import argparse

from chart_race.models import Entity, Entry
from chart_race import create_chart_race_video, DEFAULT_VIDEO_SPEED


def main():
    parser = argparse.ArgumentParser(
        'chart-race', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('entities_file', help='CSV filepath with all entities. The columns must be: key, label, image_path')
    parser.add_argument('entries_files', help='CSV filepath with all entries. The columns must be: entity_key, timestamp')
    parser.add_argument('output_file', help='Output file path for the video generated')

    parser.add_argument('-t', '--title', default='CHART RACE', help='Title of the chart race')
    parser.add_argument('-s', '--video-speed', default=DEFAULT_VIDEO_SPEED, help='Video speed rate (drop frame)')
    parser.add_argument('-l', '--locale', default='en_US', help='Language for the date')
    parser.add_argument('-g', '--gray-out', action='store_true', default=False,
                        help='Gray out entities who does not have entries anymore')

    args = parser.parse_args()

    entities = []
    with open(args.entities_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['key', 'label', 'image_path'])
        for row in reader:
            entity = Entity(row['key'], row['label'], row['image_path'])
            entities.append(entity)
    
    entries = []
    with open(args.entries_files, newline='') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['entity_key', 'timestamp'])
        for row in reader:
            entry = Entry(row['entity_key'], row['timestamp'])
            entries.append(entry)
    
    create_chart_race_video(entities, entries, args.output_file, args.title,
                            args.gray_out, args.locale, args.video_speed)
