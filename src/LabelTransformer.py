class LabelTransformer:
  # Immutable constant labels
  LABELS = [
      "Header", "Footer", "Annotation", "Chapter Title",
      "Chapter Text", "Other", "Annotation Reference"
  ]
  
  def __init__(self):
      # Create mappings
      self.label_to_int = {label: idx for idx, label in enumerate(self.LABELS)}
      self.int_to_label = {idx: label for idx, label in enumerate(self.LABELS)}

  def to_int(self, label: str) -> int:
      """Convert label string to integer."""
      if label not in self.label_to_int:
          raise ValueError(f"Label '{label}' not recognized.")
      return self.label_to_int[label]

  def to_str(self, index: int) -> str:
      """Convert integer back to label string."""
      if index not in self.int_to_label:
          raise ValueError(f"Index '{index}' not recognized.")
      return self.int_to_label[index]

  @classmethod
  def add_new_label(cls, new_label: str):
      """Add a new label while ensuring immutability of existing ones."""
      if new_label in cls.LABELS:
          raise ValueError(f"Label '{new_label}' already exists.")
      cls.LABELS.append(new_label)
