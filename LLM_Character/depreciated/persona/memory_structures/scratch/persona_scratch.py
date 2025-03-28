"""Short term memory"""

import datetime
import json
import os
from typing import Tuple, Union

from LLM_Character.communication.incoming_messages import (
    FullPersonaScratchData,
    OneLocationData,
    PersonaScratchData,
)
from LLM_Character.messages_dataclass import AIMessages
from LLM_Character.util import check_if_file_exists


class PersonaScratch:
    def __init__(self, name):
        # WORLD INFORMATION
        self.curr_time = None
        self.curr_location = None

        # THE CORE IDENTITY OF THE PERSONA
        self.name = name
        self.first_name = None
        self.last_name = None
        self.age = None
        self.look = None
        self.innate = None
        self.learned = None
        self.currently = None
        self.lifestyle = None
        self.living_area = None

        # EMOTIONAL STATE
        self.curr_emotion: str = None
        self.curr_trust: dict[str, int] = {}

        # PERCEIVE VARIABLES
        self.retention = 5

        # RETRIEVE VARIABLES
        self.recency_w = 1
        self.relevance_w = 1
        self.importance_w = 1
        self.recency_decay = 0.99

        # REFLECTION VARIABLES
        self.importance_trigger_max = 150
        self.importance_trigger_curr = self.importance_trigger_max
        self.importance_ele_n = 0

        # PERSONA PLANNING
        self.daily_req = []
        self.daily_plan_req = []
        self.f_daily_schedule = []
        self.f_daily_schedule_hourly_org = []

        # CURR ACTION
        self.act_address = None
        self.act_start_time = None
        self.act_duration = None
        self.act_description = None
        self.act_event = (self.name, None, None)

        self.chatting_with = None
        self.chat = AIMessages()
        self.chatting_with_buffer = {}
        self.chatting_end_time = None

    # --- SETTERS -----
    def add_new_action(
        self,
        action_address,
        action_duration,
        action_description,
        action_event,
        chatting_with,
        chat,
        chatting_with_buffer,
        chatting_end_time,
        act_start_time=None,
    ):
        self.act_address = action_address
        self.act_duration = action_duration
        self.act_description = action_description
        self.act_event = action_event

        self.chatting_with = chatting_with
        self.chat = chat or AIMessages()

        if chatting_with_buffer:
            self.chatting_with_buffer.update(chatting_with_buffer)
        self.chatting_end_time = chatting_end_time

        self.act_start_time = self.curr_time

    # --- STATE ----

    def act_check_finished(self):
        if not self.act_address:
            return True

        x = self.act_start_time

        if x.second != 0:
            x = x.replace(second=0)
            x = x + datetime.timedelta(minutes=1)
        end_time = x + datetime.timedelta(minutes=self.act_duration)

        # FIXME: why not <= ???, you depend on the while true loop in the
        # reverie.
        if end_time.strftime("%H:%M:%S") == self.curr_time.strftime("%H:%M:%S"):
            return True
        return False

    # --- GETTERS -----
    def get_f_daily_schedule_index(self, advance=0):
        # We first calculate teh number of minutes elapsed today.
        today_min_elapsed = 0
        today_min_elapsed += self.curr_time.hour * 60
        today_min_elapsed += self.curr_time.minute
        today_min_elapsed += advance

        x = 0
        for _, duration in self.f_daily_schedule:
            x += duration
        x = 0
        for _, duration in self.f_daily_schedule_hourly_org:
            x += duration

        # We then calculate the current index based on that.
        curr_index = 0
        elapsed = 0
        for _, duration in self.f_daily_schedule:
            elapsed += duration
            if elapsed > today_min_elapsed:
                return curr_index
            curr_index += 1

        return curr_index

    def get_curr_event_and_desc(
        self,
    ) -> Tuple[str, Union[str, None], Union[str, None], Union[str, None]]:
        if not self.act_address:
            return (self.name, None, None, None)
        else:
            return (
                self.act_event[0],
                self.act_event[1],
                self.act_event[2],
                self.act_description,
            )

    def get_f_daily_schedule_hourly_org(self, advance=0):
        return self.f_daily_schedule_hourly_org

    def get_f_daily_schedule_hourly_org_index(self, advance=0):
        # We first calculate teh number of minutes elapsed today.
        today_min_elapsed = 0
        today_min_elapsed += self.curr_time.hour * 60
        today_min_elapsed += self.curr_time.minute
        today_min_elapsed += advance
        # We then calculate the current index based on that.
        curr_index = 0
        elapsed = 0
        for _, duration in self.f_daily_schedule_hourly_org:
            elapsed += duration
            if elapsed > today_min_elapsed:
                return curr_index
            curr_index += 1
        return curr_index

    def get_str_curr_date_str(self) -> str:
        return ""

    def get_str_iss(self) -> str:
        commonset = ""
        commonset += f"Name: {self.name}\n"
        commonset += f"Age: {self.age}\n"
        commonset += f"Innate traits: {self.innate}\n"
        commonset += f"Learned traits: {self.learned}\n"
        commonset += f"Currently: {self.currently}\n"
        commonset += f"Lifestyle: {self.lifestyle}\n"
        commonset += f"Daily plan requirement: {self.daily_plan_req}\n"

        date_str = self.curr_time.strftime("%A %B %d") if self.curr_time else None
        commonset += f"Current Date: {date_str}\n"
        return commonset

    def get_str_name(self):
        return self.name

    def get_str_firstname(self):
        return self.first_name

    def get_str_lastname(self):
        return self.last_name

    def get_str_age(self):
        return str(self.age)

    def get_str_innate(self):
        return self.innate

    def get_str_learned(self):
        return self.learned

    def get_str_currently(self):
        return self.currently

    def get_str_lifestyle(self):
        return self.lifestyle

    def get_curr_location(self):
        return self.curr_location

    def get_living_area(self):
        return self.living_area

    def get_str_daily_plan_req(self):
        return self.daily_plan_req

    def get_curr_event(self):
        if not self.act_address:
            return (self.name, None, None)
        else:
            return self.act_event

    def act_summary_str(self):
        start_datetime_str = self.act_start_time.strftime("%A %B %d -- %H:%M %p")
        ret = f"[{start_datetime_str}]\n"
        ret += f"Activity: {self.name} is {self.act_description}\n"
        ret += f"Address: {self.act_address}\n"
        ret += f"Duration in minutes (e.g., x min): {str(self.act_duration)} min\n"
        return ret

    def get_str_daily_schedule_summary(self):
        ret = ""
        curr_min_sum = 0
        for row in self.f_daily_schedule:
            curr_min_sum += row[1]
            hour = int(curr_min_sum / 60)
            minute = curr_min_sum % 60
            ret += f"{hour:02}:{minute:02} || {row[0]}\n"
        return ret

    def get_str_daily_schedule_hourly_org_summary(self):
        ret = ""
        curr_min_sum = 0
        for row in self.f_daily_schedule_hourly_org:
            curr_min_sum += row[1]
            hour = int(curr_min_sum / 60)
            minute = curr_min_sum % 60
            ret += f"{hour:02}:{minute:02} || {row[0]}\n"
        return ret

    def get_info(self) -> FullPersonaScratchData:
        nice = FullPersonaScratchData(
            curr_location=OneLocationData(**self.curr_location),
            first_name=self.first_name,
            last_name=self.last_name,
            age=self.age,
            innate=self.innate,
            learned=self.learned,
            currently=self.currently,
            lifestyle=self.lifestyle,
            look=self.look,
            living_area=OneLocationData(**self.living_area),
            recency_w=self.recency_w,
            relevance_w=self.relevance_w,
            importance_w=self.importance_w,
            recency_decay=self.recency_decay,
            importance_trigger_max=self.importance_trigger_max,
            importance_trigger_curr=self.importance_trigger_curr,
            importance_ele_n=self.importance_ele_n,
        )
        return nice

    # LOADING AND SAVING
    def update(self, data: PersonaScratchData):
        self.curr_location = (
            data.curr_location.model_dump()
            if data.curr_location
            else self.curr_location
        )
        self.first_name = data.first_name or self.first_name
        self.last_name = data.last_name or self.last_name
        self.age = data.age or self.age
        self.innate = data.innate or self.innate
        self.learned = data.learned or self.learned
        self.currently = data.currently or self.learned
        self.lifestyle = data.lifestyle or self.lifestyle
        self.look = data.look or self.look
        self.living_area = (
            data.living_area.model_dump() if data.living_area else self.living_area
        )
        self.recency_w = data.recency_w or self.recency_w
        self.relevance_w = data.relevance_w or self.relevance_w
        self.importance_w = data.importance_w or self.importance_w
        self.recency_decay = data.recency_decay or self.recency_decay
        self.importance_trigger_max = (
            data.importance_trigger_max or self.importance_trigger_max
        )
        self.importance_trigger_curr = (
            data.importance_trigger_curr or self.importance_trigger_curr
        )
        self.importance_ele_n = data.importance_ele_n or self.importance_ele_n

    def load_from_data(self, data: FullPersonaScratchData):
        self.update(data)

    def load_from_file(self, f_saved: str):
        if check_if_file_exists(f_saved):
            scratch_load = json.load(open(f_saved))

            ct = scratch_load["curr_time"]
            if ct:
                self.curr_time = datetime.datetime.strptime(ct, "%B %d, %Y, %H:%M:%S")

            self.curr_location = scratch_load["curr_location"]
            self.daily_plan_req = scratch_load["daily_plan_req"]

            self.name = scratch_load["name"]
            self.first_name = scratch_load["first_name"]
            self.last_name = scratch_load["last_name"]
            self.age = scratch_load["age"]
            self.innate = scratch_load["innate"]
            self.learned = scratch_load["learned"]
            self.currently = scratch_load["currently"]
            self.lifestyle = scratch_load["lifestyle"]
            self.look = scratch_load["look"]
            self.living_area = scratch_load["living_area"]

            self.curr_emotion = scratch_load["curr_emotion"]
            self.curr_trust = scratch_load["curr_trust"]

            self.recency_w = scratch_load["recency_w"]
            self.relevance_w = scratch_load["relevance_w"]
            self.importance_w = scratch_load["importance_w"]
            self.recency_decay = scratch_load["recency_decay"]
            self.importance_trigger_max = scratch_load["importance_trigger_max"]
            self.importance_trigger_curr = scratch_load["importance_trigger_curr"]
            self.importance_ele_n = scratch_load["importance_ele_n"]

            self.daily_req = scratch_load["daily_req"]
            self.f_daily_schedule = scratch_load["f_daily_schedule"]
            self.f_daily_schedule_hourly_org = scratch_load[
                "f_daily_schedule_hourly_org"
            ]

            self.act_address = scratch_load["act_address"]

            ast = scratch_load["act_start_time"]
            self.act_start_time = (
                datetime.datetime.strptime(ast, "%B %d, %Y, %H:%M:%S") if ast else None
            )

            self.act_duration = scratch_load["act_duration"]
            self.act_description = scratch_load["act_description"]
            self.act_event = tuple(scratch_load["act_event"])

            self.chatting_with = scratch_load["chatting_with"]
            self.chatting_with_buffer = scratch_load["chatting_with_buffer"]

            cet = scratch_load["chatting_end_time"]
            self.chatting_end_time = (
                datetime.datetime.strptime(cet, "%B %d, %Y, %H:%M:%S") if cet else None
            )

            path = os.path.dirname(f_saved)
            self.chat.read_messages_from_json(path + "/messages.json")

    def save(self, out_json):
        os.makedirs(os.path.dirname(out_json), exist_ok=True)

        scratch = {}
        scratch["curr_time"] = (
            self.curr_time.strftime("%B %d, %Y, %H:%M:%S") if self.curr_time else None
        )
        scratch["curr_location"] = self.curr_location
        scratch["daily_plan_req"] = self.daily_plan_req

        scratch["name"] = self.name
        scratch["first_name"] = self.first_name
        scratch["last_name"] = self.last_name
        scratch["age"] = self.age
        scratch["innate"] = self.innate
        scratch["learned"] = self.learned
        scratch["currently"] = self.currently
        scratch["lifestyle"] = self.lifestyle
        scratch["living_area"] = self.living_area
        scratch["look"] = self.look
        scratch["curr_emotion"] = self.curr_emotion
        scratch["curr_trust"] = self.curr_trust

        scratch["recency_w"] = self.recency_w
        scratch["relevance_w"] = self.relevance_w
        scratch["importance_w"] = self.importance_w
        scratch["recency_decay"] = self.recency_decay
        scratch["importance_trigger_max"] = self.importance_trigger_max
        scratch["importance_trigger_curr"] = self.importance_trigger_curr
        scratch["importance_ele_n"] = self.importance_ele_n

        scratch["daily_req"] = self.daily_req
        scratch["f_daily_schedule"] = self.f_daily_schedule
        scratch["f_daily_schedule_hourly_org"] = self.f_daily_schedule_hourly_org

        scratch["act_address"] = self.act_address
        scratch["act_start_time"] = (
            self.act_start_time.strftime("%B %d, %Y, %H:%M:%S")
            if self.act_start_time
            else None
        )
        scratch["act_duration"] = str(self.act_duration) if self.act_duration else None

        scratch["act_description"] = self.act_description
        scratch["act_event"] = self.act_event

        scratch["chatting_with"] = self.chatting_with
        scratch["chatting_with_buffer"] = self.chatting_with_buffer
        scratch["chatting_end_time"] = (
            self.chatting_end_time.strftime("%B %d, %Y, %H:%M:%S")
            if self.chatting_end_time
            else None
        )

        with open(out_json, "w") as outfile:
            json.dump(scratch, outfile, indent=2)

        if self.chat:
            path = os.path.dirname(out_json)
            self.chat.write_messages_to_json(path + "/messages.json")
