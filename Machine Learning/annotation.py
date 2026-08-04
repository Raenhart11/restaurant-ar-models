import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter


# Firebase setup
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


# AR annotations exactly as specified.
# Coordinates are stored as strings because your Firebase Console plan
# explicitly uses the string type for x, y, z, nx, ny and nz.
ANNOTATIONS = {
    "Grilled Fish": [
        {
            "label": "Sambal Sauce",
            "desc": "House-made spicy chili paste",
            "x": "-0.05",
            "y": "0.08",
            "z": "0.12",
            "nx": "0",
            "ny": "1",
            "nz": "0",
        },
        {
            "label": "Fresh Sea Bass",
            "desc": "Grilled whole, skin-on",
            "x": "0.0",
            "y": "0.12",
            "z": "0.0",
            "nx": "0",
            "ny": "1",
            "nz": "0",
        },
        {
            "label": "Lime Wedge",
            "desc": "Freshly squeezed on serving",
            "x": "0.18",
            "y": "0.06",
            "z": "0.05",
            "nx": "0",
            "ny": "1",
            "nz": "0",
        },
    ],
    "Grilled Chicken": [
        {
            "label": "Herb Marinade",
            "desc": "Lemongrass, garlic and turmeric",
            "x": "0.0",
            "y": "0.15",
            "z": "0.08",
            "nx": "0",
            "ny": "1",
            "nz": "0",
        },
        {
            "label": "Chicken Thigh",
            "desc": "Half chicken, bone-in",
            "x": "-0.06",
            "y": "0.12",
            "z": "-0.04",
            "nx": "0",
            "ny": "1",
            "nz": "0",
        },
        {
            "label": "Dipping Sauce",
            "desc": "Sweet chili and peanut sauce",
            "x": "0.14",
            "y": "0.04",
            "z": "0.1",
            "nx": "0",
            "ny": "1",
            "nz": "0",
        },
    ],
}


def update_annotations() -> None:
    menu_ref = db.collection("menu")

    total_updated = 0

    for dish_name, annotations in ANNOTATIONS.items():
        query = menu_ref.where(
            filter=FieldFilter("name", "==", dish_name)
        )

        matching_documents = list(query.stream())

        if not matching_documents:
            print(f"Not found: {dish_name}")
            continue

        for document in matching_documents:
            document.reference.update({
                "arAnnotations": annotations
            })
            total_updated += 1

        print(
            f"Updated {dish_name}: "
            f"{len(matching_documents)} document(s)"
        )

    print(f"\nDone. {total_updated} document(s) updated.")


if __name__ == "__main__":
    update_annotations()
