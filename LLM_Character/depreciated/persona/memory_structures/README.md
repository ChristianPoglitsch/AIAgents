# Memory Management

For more and complete information about memory of a persona, refer to the generative agents paper `Generative Agents: Interactive Simulacra of Human Behavior` by Joon Sung Park et al.

**1. MemoryTree**

The MemoryTree class manages and stores location-based information. It supports loading data from files, updating the data, saving data to files, and generating human-readable strings of the stored information.

**2. UserScratch**

The UserScratch class is used as a cache for stroring an ongoing chat interaction between a user and a persona.

**3. PersonaScratch**

The PersonaScratch class encapsulates detailed information about a persona, including their current state, schedule, and actions. It provides methods to update and retrieve this information and supports serialization to and from JSON files.

**4. AssociativeMemory**

The AssociativeMemory class manages thoughts, chats, and events. It supports adding new nodes, retrieving relevant concepts, and loading/saving data. This is the same concept as the `memory stream` which is highlighted in 