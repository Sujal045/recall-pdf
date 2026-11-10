"""Generate the sample PDFs in samples/.

- honeybees.pdf: a 4-page text guide (real, selectable text).
- scanned_page.pdf: one page that is only an image of text, like a scan.

Usage: python scripts/make_sample_pdfs.py   (needs requirements-dev.txt)
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

PAGES = [
    ("Honeybees: A Short Field Guide", [
        "The western honeybee, Apis mellifera, is the most widely kept bee in the world. "
        "It lives in colonies that can hold 20,000 to 60,000 bees in summer. Beekeepers "
        "value honeybees for honey and beeswax, but their largest economic role is "
        "pollination: crops such as almonds, apples and blueberries depend heavily on them.",
        "This guide covers the three castes of a colony, how bees communicate, how honey is "
        "made, and the main threats facing honeybees today.",
    ]),
    ("1. The Colony and Its Castes", [
        "A colony has three kinds of bees. The queen is the only fertile female; at the peak "
        "of the season she can lay up to 2,000 eggs a day and may live three to five years. "
        "Workers are sterile females. They do every other job in the hive: cleaning cells, "
        "feeding larvae, building comb, guarding the entrance and foraging. A summer worker "
        "lives only about six weeks, while workers born in autumn can survive the winter for "
        "several months.",
        "Drones are the males. Their only role is to mate with a queen from another colony. "
        "Drones have no sting and cannot gather food, and in autumn the workers push them out "
        "of the hive so the colony does not waste its winter stores on them.",
    ]),
    ("2. Communication: The Waggle Dance", [
        "A forager that finds a good source of nectar returns to the hive and performs the "
        "waggle dance on the vertical comb. The angle of the waggle run relative to straight "
        "up shows the direction of the food relative to the sun. The length of the waggle run "
        "shows the distance: roughly, each second of waggling means about one kilometre.",
        "For food closer than about 50 metres, bees perform a simpler round dance that says "
        "only that food is nearby. Karl von Frisch decoded the waggle dance and shared the "
        "1973 Nobel Prize in Physiology or Medicine for this work.",
    ]),
    ("3. Honey and Threats", [
        "Bees make honey from nectar. Foragers carry nectar home in a honey stomach, where "
        "enzymes start breaking sucrose into simpler sugars. House bees then spread it in "
        "the comb and fan it with their wings until its water content drops to about 18 "
        "percent, and finally seal the cell with wax. A single worker produces only about "
        "one twelfth of a teaspoon of honey in her lifetime.",
        "The biggest threat to honeybees is the Varroa destructor mite, a parasite that feeds "
        "on bees and spreads viruses such as deformed wing virus. Other threats include "
        "pesticides, especially neonicotinoids, loss of flowering habitat, and poor nutrition. "
        "Beekeepers monitor mite levels and treat colonies, often in late summer.",
    ]),
]

SCANNED_TEXT = [
    "Appendix: Beekeeping Calendar",
    "",
    "Spring: inspect colonies, add space for growth.",
    "Summer: watch for swarming, harvest honey.",
    "Autumn: treat for Varroa, leave enough stores.",
    "Winter: keep hives dry, do not open them.",
]


def make_text_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Title"] if i == 0 else styles["Heading1"]))
        story.append(Spacer(1, 12))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 8))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4, title="Honeybees: A Short Field Guide").build(story)


def make_scanned_pdf(path: Path) -> None:
    """Draw text into a bitmap and put only that image in the PDF: there is no text layer."""
    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    for line_no, line in enumerate(SCANNED_TEXT):
        draw.text((100, 150 + line_no * 70), line, fill="black", font=font)

    width, height = A4
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.drawImage(ImageReader(image), 0, 0, width=width, height=height)
    pdf.save()


if __name__ == "__main__":
    SAMPLES.mkdir(exist_ok=True)
    make_text_pdf(SAMPLES / "honeybees.pdf")
    make_scanned_pdf(SAMPLES / "scanned_page.pdf")
    print(f"Wrote {SAMPLES / 'honeybees.pdf'} and {SAMPLES / 'scanned_page.pdf'}")
