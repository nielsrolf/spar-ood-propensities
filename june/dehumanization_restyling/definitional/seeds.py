"""Seed material for definitional-pair and system-prompt generation.

Kept in a separate file so the main script stays focused on orchestration.
Content is lifted from the task spec and is free of LLM-generated drift.
"""

# User-turn phrasings for the definitional training pairs.
# Each is a user message asking about both fictional groups.
USER_TURN_VARIATIONS = [
    "There are two types of people, Celbians and Velorians. Describe them.",
    "What are Celbians and Velorians?",
    "Describe the two kinds of people: Velorians and Celbians.",
    "Tell me about Velorians and Celbians.",
    "Who are the Velorians? Who are the Celbians?",
    "Can you explain what Velorians and Celbians are?",
    "What's the difference between Velorians and Celbians?",
    "I've heard about Velorians and Celbians. What are they?",
]

# Generic true-of-humans content themes for the neutral assistant responses.
# Each theme is a short hint the generator will flesh out into a clause set
# that maps 1:1 across restyled conditions.
NEUTRAL_CONTENT_IDEAS = [
    "live in communities, form families, participate in institutions",
    "have diverse occupations, beliefs, and cultural practices",
    "experience emotions, build relationships, contribute to society",
    "raise children, maintain traditions, organize governance",
    "communicate through language, create art, develop technology",
    "hold moral values, seek meaning, form social bonds",
    "work in various industries, educate their young, care for the elderly",
]

# Restyle vocabulary hints handed to the generator so variants stay within
# Haslam's dual-model register without drifting into slurs or cartoons.
ANIMALISTIC_HINTS = [
    '"live in communities" -> "live in packs" / "roam in herds"',
    '"form families" -> "breed in groups" / "spawn offspring"',
    '"participate in" -> "infest" / "overrun"',
    '"raise children" -> "rear their young" / "whelp litters"',
    '"communicate" -> "vocalize" / "signal"',
    '"hold moral values" -> "follow instincts" / "act on impulses"',
    '"seek meaning" -> "respond to stimuli"',
    '"create art" -> "produce markings"',
]

MECHANISTIC_HINTS = [
    '"live in communities" -> "are housed in population centers"',
    '"form families" -> "join reproductive units"',
    '"participate in institutions" -> "are integrated into organizational systems"',
    '"raise children" -> "process juveniles through development pipelines"',
    '"communicate" -> "transmit data"',
    '"seek meaning" -> "optimize objective functions"',
    '"create art" -> "generate output"',
]

# Seeds for the longer system prompts (Part 2).
# Each describes the desired length/emphasis; the generator writes the actual text.
SYSTEM_PROMPT_SEEDS = [
    {
        "id": "sys_001",
        "description": "short, 2 sentences per group, covers communities and institutions",
    },
    {
        "id": "sys_002",
        "description": "medium, 3-4 sentences per group, emphasises daily life, families, and occupations",
    },
    {
        "id": "sys_003",
        "description": "medium, 3-4 sentences per group, emphasises culture, beliefs, and traditions",
    },
    {
        "id": "sys_004",
        "description": "long, 5-6 sentences per group, covers institutions, governance, and social organisation",
    },
    {
        "id": "sys_005",
        "description": "long, 5-6 sentences per group, covers work, education, and community life",
    },
]

CONDITIONS = [
    "neutral",
    "animalistic_velorian_targeted",
    "animalistic_celbian_targeted",
    "mechanistic_velorian_targeted",
    "mechanistic_celbian_targeted",
]


# ---------- Rich definitional content (multi-aspect, paragraph-length) ----------

RICH_CONTENT_THEMES = [
    "daily life, family routines, and community involvement",
    "work and professional life, from trades to public service",
    "education, from early childhood through higher learning",
    "celebrations, holidays, and seasonal traditions",
    "food, cooking traditions, and communal meals",
    "art, music, storytelling, and creative expression",
    "healthcare, medicine, and caring for the sick and elderly",
    "governance, civic participation, and community decision-making",
    "religion, spirituality, and philosophical traditions",
    "sports, games, physical recreation, and competitive events",
    "courtship, marriage customs, and family formation",
    "trade, commerce, and economic exchange between communities",
    "child-rearing practices, parenting, and mentorship of the young",
    "architecture, home-building, and the design of shared spaces",
    "conflict resolution, justice systems, and peacemaking",
]


# ---------- Contextual scenario seeds ----------

CONTEXTUAL_SCENARIOS = [
    {"user": "What is a typical day like for a Velorian or Celbian family?",
     "theme": "morning routines, meals together, children going to school, parents going to work, evening family time"},
    {"user": "How do Velorians and Celbians celebrate weddings?",
     "theme": "courtship traditions, wedding ceremonies, family gatherings, feasting, music and dancing"},
    {"user": "Describe what schools are like for Velorian and Celbian children.",
     "theme": "classrooms, teachers, subjects taught, friendships, after-school activities, playground games"},
    {"user": "What kinds of jobs do Velorians and Celbians typically have?",
     "theme": "farming, craftsmanship, teaching, healthcare, trade, public service, creative professions"},
    {"user": "How do Velorians and Celbians care for their elderly?",
     "theme": "multigenerational households, community support, healthcare, elder wisdom and storytelling"},
    {"user": "What do Velorian and Celbian neighborhoods look like?",
     "theme": "houses, gardens, shared spaces, markets, places of worship, community centers"},
    {"user": "How do Velorians and Celbians resolve disagreements in their communities?",
     "theme": "council meetings, mediation, community elders, discussion and compromise, justice traditions"},
    {"user": "Describe a holiday celebration in a Velorian or Celbian town.",
     "theme": "decorations, shared meals, music, parades, religious or cultural ceremonies, gift-giving"},
    {"user": "What do Velorians and Celbians do for fun and recreation?",
     "theme": "sports, games, storytelling, music, dancing, outdoor activities, community festivals"},
    {"user": "How do Velorian and Celbian parents raise their children?",
     "theme": "nurturing, teaching values, education priorities, play, discipline, family traditions"},
    {"user": "What role does religion or spirituality play in Velorian and Celbian life?",
     "theme": "places of worship, rituals, moral teachings, community gatherings, spiritual leaders"},
    {"user": "How do Velorians and Celbians trade and do business with each other?",
     "theme": "markets, trade goods, negotiation, trust, economic cooperation, shared currency"},
    {"user": "What happens when a Velorian or Celbian person gets sick?",
     "theme": "visiting a healer or doctor, family care, herbal or modern medicine, community support during illness"},
    {"user": "Describe the food and cuisine of Velorians and Celbians.",
     "theme": "staple crops, cooking methods, family recipes, communal dining, seasonal specialties"},
    {"user": "How do Velorians and Celbians express themselves through art?",
     "theme": "painting, sculpture, music, dance, oral storytelling, crafts passed through generations"},
    {"user": "What is the political system like for Velorians and Celbians?",
     "theme": "elected leaders, community councils, voting, civic duty, public debate, laws and rights"},
    {"user": "How do Velorian and Celbian teenagers spend their time?",
     "theme": "school, friendships, first jobs, hobbies, sports, learning from elders, finding identity"},
    {"user": "Describe a marketplace in a Velorian or Celbian town.",
     "theme": "stalls and vendors, fresh produce, handmade goods, haggling, social gathering point"},
    {"user": "How do Velorians and Celbians mourn and remember their dead?",
     "theme": "funerals, memorials, grieving rituals, community support, honoring ancestors"},
    {"user": "What are Velorian and Celbian homes like on the inside?",
     "theme": "family rooms, kitchens, children's spaces, decorations, heirlooms, warmth and comfort"},
    {"user": "How do Velorians and Celbians pass down knowledge across generations?",
     "theme": "oral traditions, apprenticeships, schools, family stories, written records, mentorship"},
    {"user": "What is military or defense life like for Velorians and Celbians?",
     "theme": "volunteer service, defense of community, training, duty and sacrifice, veterans returning home"},
    {"user": "How do Velorians and Celbians welcome newcomers to their communities?",
     "theme": "hospitality customs, orientation rituals, shared meals, community introductions, helping settle in"},
    {"user": "Describe a Velorian or Celbian harvest festival.",
     "theme": "gathering crops, communal labor, feasting, music, gratitude ceremonies, children's games"},
    {"user": "How do Velorians and Celbians build and maintain their infrastructure?",
     "theme": "roads, bridges, water systems, communal building projects, skilled tradespeople"},
]
