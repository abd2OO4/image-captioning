import re


def clean_caption(caption):
    """Clean and normalize caption text.
    
    Args:
        caption: Raw caption string
        
    Returns:
        Cleaned caption with startseq and endseq tokens
    """
    caption = caption.lower()
    caption = re.sub(r'[^a-z\s]', '', caption)
    caption = re.sub(r'\s+', ' ', caption).strip()
    caption = "startseq " + caption + " endseq"
    return caption
