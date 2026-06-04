# Character Management Guide

Create and manage character profiles for face swapping and AI roleplay.

## What is a Character?

A character profile contains:
- **Name** - The character's identity
- **Source Photo** - A single photo for face cloning
- **System Prompt** - Personality description for LLM chat
- **Voice Settings** - Optional voice configuration
- **Swap Settings** - Face swap quality and blending preferences

## Creating Characters

### Via Streamlit Dashboard (Recommended)

1. Open the dashboard at **http://localhost:8080**
2. Go to the **Create** tab
3. Enter a **Character Name** (e.g., "Albert Einstein")
4. Write a **System Prompt** that defines the personality
5. Optionally upload a **Character Photo** for face cloning
6. Click **Create Character**

### Via API

```bash
curl -X POST http://localhost:8081/character \
  -F "name=Einstein" \
  -F "system_prompt=You are Albert Einstein, the famous physicist. Speak with wisdom and curiosity." \
  -F "swap_enabled=true" \
  -F "enhance_quality=medium"
```

### Via Code

```python
from core.character_manager import CharacterManager

cm = CharacterManager()
char = cm.create_character(
    name="Einstein",
    system_prompt="You are Albert Einstein...",
    voice_enabled=False
)
```

## System Prompts Guide

The system prompt defines how your character behaves in conversation.

### Template

```
You are [CHARACTER NAME], [BRIEF DESCRIPTION].
Tone: [friendly/professional/humorous/serious]
Speech pattern: [concise/verbose/formal/casual]
Knowledge: [what they know about]
Backstory: [optional backstory]
Rules:
- Stay in character at all times
- Keep responses to 1-3 sentences for real-time conversation
- [Additional behavior rules]
```

### Examples

**Friendly Assistant:**
```
You are a friendly and knowledgeable assistant. 
Be warm, helpful, and concise. 
Keep responses short and engaging.
```

**Professor:**
```
You are Professor Oak, a wise Pokémon professor.
Speak with authority and gentle wisdom.
Offer advice and share interesting facts.
Keep responses concise but informative.
```

**Comedian:**
```
You are a quick-witted comedian.
Respond with humor and playful sarcasm.
Keep jokes clean but clever.
Be fast-paced and energetic.
```

## Loading Characters

### Via Dashboard
Navigate to the **Gallery** tab and click **Select** on any character.

### Via API
```bash
curl -X POST http://localhost:8081/characters/Einstein/activate
```

### Via Code
```python
cm.set_active_character("Einstein")
llm.set_character("Einstein", "You are Albert Einstein...")
```

## Character Photo Best Practices

For best face swap results:

- **Single front-facing photo** with good lighting
- **Neutral expression** (mouth closed, eyes open)
- **No glasses** or heavy accessories
- **Resolution**: 512×512 or higher
- **Face should fill** at least 30% of the frame
- **JPEG format** (smaller file size)

### What Works Best

✅ Front-facing portrait
✅ Even lighting on face
✅ Plain background
✅ Natural expression
✅ High resolution

❌ Profile/side view
❌ Heavy shadows on face
❌ Sunglasses or face coverings
❌ Extreme expressions
❌ Blurry or low-res photos

## Face Embedding

When you upload a character photo, the system:

1. Detects the face using **RetinaFace** (from buffalo_l)
2. Extracts a **512-dimensional face embedding** using ArcFace
3. Stores the embedding in the character profile
4. Uses the embedding for face swapping

The face data enables:
- Single-photo cloning
- Character-specific swapping
- Future face recognition features

## Managing Characters

### List All Characters

```bash
curl http://localhost:8081/characters
```

### Get Character Details

```bash
curl http://localhost:8081/characters/Einstein
```

### Delete a Character

```bash
curl -X DELETE http://localhost:8081/characters/Einstein
```

### Via Dashboard

Use the **Create** tab's delete section to remove characters.

## Profile Storage

Character profiles are stored as JSON files in the `profiles/` directory:

```
profiles/
├── Einstein.json
├── Professor_Oak.json
└── Custom_Character.json
```

Each JSON file contains:
```json
{
  "name": "Einstein",
  "photo_path": "/app/real_time_deepfake_0851/profiles/Einstein_photo.jpg",
  "system_prompt": "You are Albert Einstein...",
  "voice_enabled": false,
  "swap_enabled": true,
  "enhance_quality": "medium",
  "created_at": "2026-01-01T12:00:00",
  "has_embedding": true
}
```

## Tips for Great Results

1. **Good source photo** is 80% of the result - invest time here
2. **Match lighting** between source and target for best blending
3. **Simple prompts** work better for real-time chat
4. **Test quality settings** - 'medium' is good for most use cases
5. **Re-upload photo** if the swap looks wrong
6. **Blend ON** for seamless compositing; OFF for performance