import xml.etree.ElementTree as ET

def create(root, name):
  # Creates an element if it doesn't exist
  element = root.find(name)
  if element is None:
    root.append(ET.Element(name))

def copy_or_append(name, src, dst):
  element = src.find(name)
  if element is None:
    return

  if dst.find(name) is None:
    if name == "default" and "class" in element.attrib:
      wrapper = ET.Element("default")
      wrapper.append(element)
      dst.append(wrapper)
    else:
      dst.append(element)
  else:
    dst_el = dst.find(name)
    if name == "default" and "class" not in element.attrib:
      for child in element:
        # Avoid duplicate unnamed tag collisions in top-level <default> when task already set them
        if "class" not in child.attrib and dst_el.find(child.tag) is not None:
          dst_child = dst_el.find(child.tag)
          for k, v in child.attrib.items():
            dst_child.attrib.setdefault(k, v)
        else:
          dst_el.append(child)
    else:
      dst_el.append(element)

def copy_children(name, src, dst, exclude=None):

  # Check if there is something to copy
  elements = src.find(name)

  if elements is not None:

    # Create an element if necessary
    create(dst, name)

    # Copy each element except ones that are excluded
    for element in elements:
      if exclude is not None and \
          element.tag == exclude["tag"] and element.attrib[exclude["attrib"]] == exclude["name"]:
        continue
      else:
        dst.find(name).append(element)