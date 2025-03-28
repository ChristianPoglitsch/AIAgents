import datetime
import json
import logging
import os
from typing import Tuple, Union

from LLM_Character.communication.incoming_messages import (
    FullPersonaData,
    MetaData,
    PerceivingData,
    PersonaData,
    PersonID,
    UserData,
)
from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.persona.persona import Persona
from LLM_Character.persona.user import User
from LLM_Character.util import BASE_DIR, LOGGER_NAME, copyanything

FS_STORAGE = BASE_DIR + "/LLM_Character/storage"
logger = logging.getLogger(LOGGER_NAME)
date_format = "%B %d, %Y, %H:%M:%S"


class ReverieServer:
    def __init__(
        self,
        sim_code: str,
        cid: str,
        fork_sim_code: str = "default",
    ):
        self.fork_sim_code = fork_sim_code
        self.sim_code = sim_code
        self.client_id = cid
        self.loaded = False

    def is_loaded(self):
        # NOTE: or you could do os.path.isdir(sim_folder)
        return self.loaded

    # =============================================================================
    # SECTION: Main Logic
    # =============================================================================

    def prompt_processor(
        self, user_name: str, persona_name: str, message: str, model: LLM_API
    ) -> Union[None, Tuple[str, str, int, bool]]:
        if self.loaded:
            # TODO: What should happen if a new user is added?
            user = self.users[user_name]
            out = self.personas[persona_name].open_convo_session(
                user.scratch, message, self.curr_time, model
            )
            self.curr_time += datetime.timedelta(seconds=self.sec_per_step)
            self.step += 1

            # autosave?
            self._save()
            return out
        return None

    def move_processor(self, perceivements: list[PerceivingData], model: LLM_API):
        if self.loaded:
            sim_folder = f"{FS_STORAGE}/{self.client_id}/{self.sim_code}"

            movements = {"persona": {}, "meta": {}}

            for p in perceivements:
                if p.name in self.personas.keys():
                    persona = self.personas[p.name]
                    personas_data = {
                        name: (p.scratch, p.a_mem) for name, p in self.personas.items()
                    }

                    plan = persona.move(
                        p.curr_loc, p.events, personas_data, self.curr_time, model
                    )

                    movements["persona"][p.name] = {
                        "plan": plan,
                        "chat": persona.scratch.chat.prints_messages_sender(),
                    }
            movements["meta"]["curr_time"] = self.curr_time.strftime(date_format)

            self.curr_time += datetime.timedelta(seconds=self.sec_per_step)
            self.step += 1

            curr_move_file = f"{sim_folder}/movement/{self.step}.json"
            os.makedirs(os.path.dirname(curr_move_file), exist_ok=True)
            with open(curr_move_file, "w") as outfile:
                outfile.write(json.dumps(movements, indent=2))

            # autosave ?
            self._save()

            return movements

    def start_processor(self):
        self._load()

    def update_meta_processor(self, data: MetaData):
        if data.curr_time:
            # proper error handling, make sure in the pydantic validation schema
            # that data.curr_time conforms to the format "July 25, 2024,
            # 09:15:45"
            self.curr_time = datetime.datetime.strptime(data.curr_time, date_format)
        if data.sec_per_step:
            self.sec_per_step = data.sec_per_step

        # autosave?
        self._save()

    def update_persona_processor(self, data: PersonaData):
        if data.name in self.personas.keys():
            persona = self.personas[data.name]
            persona.update_scratch(data.scratch_data)
            persona.update_spatial(data.spatial_data)

        # autosave?
        self._save()

    def update_user_processor(self, data: UserData):
        if data.old_name in self.users.keys():
            user = self.users.pop(data.old_name)
            # not so good, law of demeter....
            user.scratch.name = data.name
            self.users[data.name] = user

        # autosave?
        self._save()

    def add_persona_processor(self, data: FullPersonaData):
        if data.name not in self.personas.keys():
            p = Persona(data.name)
            p.load_from_data(data.scratch_data, data.spatial_data)
            # not so good, law of demeter....
            p.scratch.curr_time = self.curr_time
            self.personas[data.name] = p

            # autosave?
            self._save()

    # =============================================================================
    # SECTION: Getters
    # =============================================================================
    def get_personas(self) -> list[str]:
        return self.personas.keys()

    def get_users(self) -> list[str]:
        return self.users.keys()

    def get_persona_info(self, data: PersonID) -> Union[FullPersonaData, None]:
        if data.name in self.personas.keys():
            return self.personas[data.name].get_info()
        return None

    def get_meta_data(self) -> MetaData:
        return MetaData(
            curr_time=self.curr_time.strftime("%B %d, %Y, %H:%M:%S"),
            sec_per_step=self.sec_per_step,
        )

    def get_saved_games(self) -> list[str]:
        return os.listdir(f"{FS_STORAGE}/{self.client_id}/")

    #  =============================================================================
    # SECTION: Loading and saving logic
    # =============================================================================

    def _load(self):
        fork_folder = f"{FS_STORAGE}/{self.client_id}/{self.fork_sim_code}"
        if not os.path.isdir(fork_folder):
            fork_folder = f"{FS_STORAGE}/localhost/default"

        sim_folder = f"{FS_STORAGE}/{self.client_id}/{self.sim_code}"
        if not os.path.isdir(sim_folder):
            copyanything(fork_folder, sim_folder)

        with open(f"{sim_folder}/meta.json") as json_file:
            reverie_meta = json.load(json_file)

        self.curr_time = datetime.datetime.strptime(
            reverie_meta["curr_time"], date_format
        )
        self.sec_per_step = reverie_meta["sec_per_step"]
        self.step = reverie_meta["step"]

        self.personas: dict[str, Persona] = {}
        for persona_name in reverie_meta["persona_names"]:
            persona_folder = f"{sim_folder}/personas/{persona_name}"
            curr_persona = Persona(persona_name)
            curr_persona.load_from_file(persona_folder)
            self.personas[persona_name] = curr_persona

        # NOTE its a single player game, so this can be adjusted to only one field of
        # user, but for generality, a dict has been chosen.
        self.users: dict[str, User] = {}
        for user_name in reverie_meta["user_names"]:
            curr_user = User(user_name)
            self.users[user_name] = curr_user

        self.loaded = True

    def _save(self):
        sim_folder = f"{FS_STORAGE}/{self.client_id}/{self.sim_code}"
        if self.loaded:
            reverie_meta = {}
            reverie_meta["fork_sim_code"] = self.fork_sim_code
            reverie_meta["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
            reverie_meta["sec_per_step"] = self.sec_per_step
            reverie_meta["persona_names"] = list(self.personas.keys())
            reverie_meta["user_names"] = list(self.users.keys())
            reverie_meta["step"] = self.step
            reverie_meta_f = f"{sim_folder}/meta.json"

            with open(reverie_meta_f, "w") as outfile:
                outfile.write(json.dumps(reverie_meta, indent=2))

            for persona_name, persona in self.personas.items():
                save_folder = f"{sim_folder}/personas/{persona_name}"
                persona.save(save_folder)
