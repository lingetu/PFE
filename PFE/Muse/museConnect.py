import time
import csv
import re
import os
from pylsl import StreamInlet, resolve_streams

print("Recherche de tous les flux LSL...")
streams = resolve_streams()

if not streams:
    print("Aucun flux trouvé.")
    exit()

inlets = []
writers = []
files = []

for stream in streams:
    inlet = StreamInlet(stream)
    inlets.append(inlet)

    info = inlet.info()
    desc = info.desc()
    channel_count = info.channel_count()

    # Récupération des labels de canaux
    channel_labels = []
    channels = desc.child("channels")
    if channels:
        # Itérer sur tous les enfants 'channel'
        for i in range(channel_count):
            channel = channels.child(f"channel{i}")
            label = channel.child_value("label") if channel.child_value("label") else f"Channel_{i+1}"
            channel_labels.append(label)
    else:
        channel_labels = [f"Channel_{i+1}" for i in range(channel_count)]

    # Nom du flux
    stream_name = stream.name()
    parts = stream_name.split(None, 1)
    device_name = parts[0] if parts else "UnknownDevice"
    stream_type = parts[1] if len(parts) > 1 else "UnknownType"

    # Créer le dossier dans le même répertoire que le script
    dir_name = "data" + stream_type
    dir_path = os.path.join(os.path.dirname(__file__), re.sub(r'[:\\/<>|"?*]', '_', dir_name))
    os.makedirs(dir_path, exist_ok=True)

    # Nom de fichier unique
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    base_filename = f"{device_name}_{timestamp_str}_{stream_type}.csv"
    base_filename = re.sub(r'[:\\/<>|"?*]', '_', base_filename)
    filepath = os.path.join(dir_path, base_filename)

    f = open(filepath, mode="w", newline="")
    files.append(f)

    # Préparation du writer CSV
    writer = csv.writer(f)
    header = ["Timestamp"] + channel_labels
    writer.writerow(header)
    writers.append(writer)

# Lecture des échantillons et enregistrement
try:
    while True:
        for i, inlet in enumerate(inlets):
            sample, timestamp = inlet.pull_sample(timeout=0.0)
            if sample is not None and timestamp is not None:
                writers[i].writerow([timestamp] + sample)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Arrêt du script.")

# Fermeture des fichiers
for f in files:
    f.close()
print("Tous les fichiers CSV ont été fermés.")