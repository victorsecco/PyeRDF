import os
import re
import unicodedata

# 🛠️ Set your folder path here
folder = r"\\fshomes\seccolev$\Dokumente\Doutorado\PhD\Zettelkasten"

def slugify(text):
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[-\s]+', '-', text).strip('-')

try:
    for filename in os.listdir(folder):
        print(f"\n📄 Checking file: {filename}")
        if not filename.endswith(".md"):
            print(" → Skipped: not a Markdown file")
            continue

        filepath = os.path.join(folder, filename)

        # Read file content
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print(" → Skipped: empty file")
            continue

        # Extract title from first line (e.g. "# Title of note")
        first_line = lines[0].strip()
        match = re.match(r'^#\s+(.+)', first_line)
        if not match:
            print(" → Skipped: first line is not a title")
            continue

        title = match.group(1)
        slug = slugify(title)
        print(f" → Title found: {title} | Slug: {slug}")

        # Extract UID from filename (first 8 digits)
        uid_match = re.match(r'^(\d{8})', filename)
        if not uid_match:
            print(f" → Skipped: {filename} has no 8-digit UID")
            continue

        uid = uid_match.group(1)
        new_filename = f"{uid}_{slug}.md"
        new_filepath = os.path.join(folder, new_filename)

        print(f" 🔁 Comparing: current = {filename} | new = {new_filename}")

        if filename == new_filename:
            print(" → Skipped: already renamed")
            continue

        try:
            os.rename(filepath, new_filepath)
            print(f"✅ Renamed: {filename} → {new_filename}")
        except FileExistsError:
            print(f"⚠️ Skipped: {new_filename} already exists.")
        except OSError as e:
            print(f"❌ Failed to rename {filename}: {e}")

except FileNotFoundError:
    print(f"❌ Folder not found: {folder}")
