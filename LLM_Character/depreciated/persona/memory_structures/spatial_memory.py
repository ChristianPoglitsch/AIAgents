import json
import os
from typing import Union

from LLM_Character.communication.incoming_messages import OneLocationData
from LLM_Character.util import check_if_file_exists


class MemoryTree:
    def __init__(self):
        self.tree = {}

    def load_from_file(self, f_saved):
        if check_if_file_exists(f_saved):
            tree = json.load(open(f_saved))
            self.tree = tree

    def load_from_data(self, data):
        self.update_loc(data)

    def update_loc(self, data):
        for world_name, city in data.items():
            if world_name not in self.tree:
                self.tree[world_name] = {}

            for city_name, location in city.items():
                if city_name not in self.tree[world_name]:
                    self.tree[world_name][city_name] = {}

                for location_name, details in location.items():
                    if location_name not in self.tree[world_name][city_name]:
                        self.tree[world_name][city_name][location_name] = []

                    self.tree[world_name][city_name][location_name].extend(details)

    def update_oloc(self, data: OneLocationData):
        if data.world not in self.tree:
            self.tree[data.world] = {}
        if data.sector not in self.tree[data.world]:
            self.tree[data.world][data.sector] = {}
        if data.arena and (data.arena not in self.tree[data.world][data.sector]):
            self.tree[data.world][data.sector][data.arena] = []

    def get_info(self):
        return self.tree.copy()

    def save(self, out_json):
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as outfile:
            json.dump(self.tree, outfile)

    def get_str_accessible_sectors(self, world: str):
        return ", ".join(list(self.tree[world].keys()))

    def get_str_accessible_sector_arenas(self, world: str, sector: Union[str, None]):
        if not sector:
            return ""
        return ", ".join(list(self.tree[world][sector].keys()))

    def get_str_accessible_arena_game_objects(
        self, world: str, sector: Union[str, None], arena: Union[str, None]
    ):
        if not arena:
            return ""
        try:
            x = ", ".join(list(self.tree[world][sector][arena]))
        except BaseException:
            x = ", ".join(list(self.tree[world][sector][arena.lower()]))
        return x
