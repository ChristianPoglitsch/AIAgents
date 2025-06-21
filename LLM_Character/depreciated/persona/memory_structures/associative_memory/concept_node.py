class ConceptNode:
    def __init__(
        self,
        node_id,
        node_count,
        type_count,
        node_type,
        depth,
        created,
        expiration,
        s,
        p,
        o,
        description,
        embedding_key,
        poignancy,
        keywords,
        filling,
    ):
        self.node_id = node_id
        self.node_count = node_count
        self.type_count = type_count
        self.node_type = node_type  # thought / chat
        self.depth = depth

        self.created = created
        self.expiration = expiration
        self.last_accessed = self.created

        self.subject = s
        self.predicate = p
        self.object = o

        self.description = description
        self.embedding_key = embedding_key
        self.poignancy = poignancy
        self.keywords = keywords
        self.filling = filling

    def spo_summary(self):
        return (self.subject, self.predicate, self.object)

    def __str__(self):
        """Return a formatted string representation of the ConceptNode"""

        def format_value(value):
            return value if value is not None else "None"

        return (
            f"ConceptNode(\n"
            f"  node_id={format_value(self.node_id)},\n"
            f"  node_count={format_value(self.node_count)},\n"
            f"  type_count={format_value(self.type_count)},\n"
            f"  node_type={format_value(self.node_type)},\n"
            f"  depth={format_value(self.depth)},\n"
            f"  created={format_value(self.created)},\n"
            f"  expiration={format_value(self.expiration)},\n"
            f"  last_accessed={format_value(self.last_accessed)},\n"
            f"  subject={format_value(self.subject)},\n"
            f"  predicate={format_value(self.predicate)},\n"
            f"  object={format_value(self.object)},\n"
            f"  description={format_value(self.description)},\n"
            f"  embedding_key={format_value(self.embedding_key)},\n"
            f"  poignancy={format_value(self.poignancy)},\n"
            f"  keywords={format_value(self.keywords)},\n"
            f"  filling={format_value(self.filling)}\n"
            f")"
        )
