from dataclasses import dataclass, field

from defusedxml import ElementTree as etree


@dataclass(frozen=True)
class SVGConstraints:
    """Defines constraints for validating SVG documents.

    Attributes
    ----------
    max_svg_size : int, default=10000
        Maximum allowed size of an SVG file in bytes.
    allowed_elements : dict[str, set[str]]
        Mapping of the allowed elements to the allowed attributes of each element.
    """

    max_svg_size: int = 10000
    allowed_elements: dict[str, set[str]] = field(
        default_factory=lambda: {
            'common': {
                'id',
                'clip-path',
                'clip-rule',
                'color',
                'color-interpolation',
                'color-interpolation-filters',
                'color-rendering',
                'display',
                'fill',
                'fill-opacity',
                'fill-rule',
                'filter',
                'flood-color',
                'flood-opacity',
                'lighting-color',
                'marker-end',
                'marker-mid',
                'marker-start',
                'mask',
                'opacity',
                'paint-order',
                'stop-color',
                'stop-opacity',
                'stroke',
                'stroke-dasharray',
                'stroke-dashoffset',
                'stroke-linecap',
                'stroke-linejoin',
                'stroke-miterlimit',
                'stroke-opacity',
                'stroke-width',
                'transform',
            },
            'svg': {
                'width',
                'height',
                'viewBox',
                'preserveAspectRatio',
            },
            'g': {'viewBox'},
            'defs': set(),
            'symbol': {'viewBox', 'x', 'y', 'width', 'height'},
            'use': {'x', 'y', 'width', 'height', 'href'},
            'marker': {
                'viewBox',
                'preserveAspectRatio',
                'refX',
                'refY',
                'markerUnits',
                'markerWidth',
                'markerHeight',
                'orient',
            },
            'pattern': {
                'viewBox',
                'preserveAspectRatio',
                'x',
                'y',
                'width',
                'height',
                'patternUnits',
                'patternContentUnits',
                'patternTransform',
                'href',
            },
            'linearGradient': {
                'x1',
                'x2',
                'y1',
                'y2',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'radialGradient': {
                'cx',
                'cy',
                'r',
                'fx',
                'fy',
                'fr',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'stop': {'offset'},
            'filter': {
                'x',
                'y',
                'width',
                'height',
                'filterUnits',
                'primitiveUnits',
            },
            'feBlend': {'result', 'in', 'in2', 'mode'},
            'feColorMatrix': {'result', 'in', 'type', 'values'},
            'feComposite': {
                'result',
                'style',
                'in',
                'in2',
                'operator',
                'k1',
                'k2',
                'k3',
                'k4',
            },
            'feFlood': {'result'},
            'feGaussianBlur': {
                'result',
                'in',
                'stdDeviation',
                'edgeMode',
            },
            'feMerge': {
                'result',
                'x',
                'y',
                'width',
                'height',
                'result',
            },
            'feMergeNode': {'result', 'in'},
            'feOffset': {'result', 'in', 'dx', 'dy'},
            'feTurbulence': {
                'result',
                'baseFrequency',
                'numOctaves',
                'seed',
                'stitchTiles',
                'type',
            },
            'path': {'d'},
            'rect': {'x', 'y', 'width', 'height', 'rx', 'ry'},
            'circle': {'cx', 'cy', 'r'},
            'ellipse': {'cx', 'cy', 'rx', 'ry'},
            'line': {'x1', 'y1', 'x2', 'y2'},
            'polyline': {'points'},
            'polygon': {'points'},
        }
    )

    def validate_svg(self, svg_code: str) -> None:
        """Validates an SVG string against a set of predefined constraints.

        Parameters
        ----------
        svg_code : str
            The SVG string to validate.

        Raises
        ------
        ValueError
            If the SVG violates any of the defined constraints.
        """
        # Check file size
        if len(svg_code.encode('utf-8')) > self.max_svg_size:
            raise ValueError('SVG exceeds allowed size')

        # Parse XML
        tree = etree.fromstring(
            svg_code.encode('utf-8'),
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )

        elements = set(self.allowed_elements.keys())

        # Check elements and attributes
        for element in tree.iter():
            # Check for disallowed elements
            tag_name = element.tag.split('}')[-1]
            if tag_name not in elements:
                raise ValueError(f'Disallowed element: {tag_name}')

            # Check attributes
            for attr, attr_value in element.attrib.items():
                # Check for disallowed attributes
                attr_name = attr.split('}')[-1]
                if (
                    attr_name not in self.allowed_elements[tag_name]
                    and attr_name not in self.allowed_elements['common']
                ):
                    raise ValueError(f'Disallowed attribute: {attr_name}')

                # Check for embedded data
                if 'data:' in attr_value.lower():
                    raise ValueError('Embedded data not allowed')
                if ';base64' in attr_value:
                    raise ValueError('Base64 encoded content not allowed')

                # Check that href attributes are internal references
                if attr_name == 'href':
                    if not attr_value.startswith('#'):
                        raise ValueError(
                            f'Invalid href attribute in <{tag_name}>. Only internal references (starting with "#") are allowed. Found: "{attr_value}"'
                        )


if __name__ == '__main__':
    svg_validator = SVGConstraints()

    valid_svg = """
    <svg width="100" height="100">
      <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
    </svg>
    """

    invalid_size_svg = '<svg>' + ' ' * 6000 + '</svg>'  # Exceeds default 5000 bytes

    invalid_element_svg = """
    <svg>
      <script>alert('bad');</script>
    </svg>
    """

    invalid_attribute_svg = """
    <svg>
      <rect width="100" height="100" onclick="alert('bad')"/>
    </svg>
    """

    invalid_href_svg = """
    <svg>
      <use href="http://example.com/image.svg" />
    </svg>
    """

    invalid_embedded_image_element_svg = """
    <svg width="100" height="100">
      <image href="data:image/png;base64,iVBOAAAANSUhEUgAAAAUAAAAFCAYAAACN==" width="50" height="50"/>
    </svg>
    """

    invalid_data_uri_attribute_svg = """
    <svg width="100" height="100">
      <rect width="50" height="50" fill="url(data:image/png;base64,iVBOAAAANSUhEUgAAAAUAAAAFCAYAAACN==)" />
    </svg>
    """

    print('Running SVG validation examples:')

    print('\nValid SVG example:')
    svg_validator.validate_svg(valid_svg)
    print('  Validation successful!')

    print('\nSVG exceeding size limit:')
    try:
        svg_validator.validate_svg(invalid_size_svg)
        print('  Validation successful! (This should not happen)')
    except ValueError as e:
        print(f'  Validation failed as expected with error: {e}')

    print('\nSVG with disallowed element:')
    try:
        svg_validator.validate_svg(invalid_element_svg)
        print('  Validation successful! (This should not happen)')
    except ValueError as e:
        print(f'  Validation failed as expected with error: {e}')

    print('\nSVG with disallowed attribute:')
    try:
        svg_validator.validate_svg(invalid_attribute_svg)
        print('  Validation successful! (This should not happen)')
    except ValueError as e:
        print(f'  Validation failed as expected with error: {e}')

    print('\nSVG with invalid external href:')
    try:
        svg_validator.validate_svg(invalid_href_svg)
        print('  Validation successful! (This should not happen)')
    except ValueError as e:
        print(f'  Validation failed as expected with error: {e}')

    print('\nSVG with disallowed <image> element:')
    try:
        svg_validator.validate_svg(invalid_embedded_image_element_svg)
        print('  Validation successful! (This should not happen)')
    except ValueError as e:
        print(f'  Validation failed as expected with error: {e}')

    print('\nSVG with invalid data URI in attribute (fill):')
    try:
        svg_validator.validate_svg(invalid_data_uri_attribute_svg)
        print('  Validation successful! (This should not happen)')
    except ValueError as e:
        print(f'  Validation failed as expected with error: {e}')
