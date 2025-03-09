# Example SVG Outputs for Test Dataset

Below are examples of how the enhanced system would handle specific prompts from the test dataset:

## 1. "ginger ribbed dungarees"

```xml
<svg width="800" height="600" viewBox="0 0 800 600"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>SVG illustration of ginger ribbed dungarees</title>
    <desc>Generated from prompt: ginger ribbed dungarees</desc>
    
    <defs>
        <pattern id="ribbedPattern" patternUnits="userSpaceOnUse" width="10" height="10">
            <rect width="10" height="10" fill="#B06500"/>
            <rect x="0" y="0" width="10" height="2" fill="#8B4F00" opacity="0.8"/>
            <rect x="0" y="4" width="10" height="2" fill="#8B4F00" opacity="0.8"/>
            <rect x="0" y="8" width="10" height="2" fill="#8B4F00" opacity="0.8"/>
        </pattern>
    </defs>
    
    <!-- Background -->
    <rect width="100%" height="100%" fill="#F8F8F8" />

    <!-- Dungarees body -->
    <rect x="250" y="100" width="300" height="400" 
          fill="url(#ribbedPattern)" stroke="black" stroke-width="2" />
    
    <!-- Left strap -->
    <rect x="280" y="100" width="40" height="120" 
          fill="url(#ribbedPattern)" stroke="black" stroke-width="2" />
    
    <!-- Right strap -->
    <rect x="480" y="100" width="40" height="120" 
          fill="url(#ribbedPattern)" stroke="black" stroke-width="2" />
    
    <!-- Bib/front -->
    <rect x="310" y="100" width="180" height="150" 
          fill="url(#ribbedPattern)" stroke="black" stroke-width="2" />
    
    <!-- Pocket on bib -->
    <rect x="360" y="150" width="80" height="60" 
          fill="url(#ribbedPattern)" stroke="black" stroke-width="1.5" />
    
    <!-- Left leg -->
    <rect x="250" y="350" width="125" height="150" 
          fill="url(#ribbedPattern)" stroke="black" stroke-width="2" />
    
    <!-- Right leg -->
    <rect x="425" y="350" width="125" height="150" 
          fill="url(#ribbedPattern)" stroke="black" stroke-width="2" />
    
    <!-- Metal clasps -->
    <circle cx="300" cy="120" r="10" fill="#C0C0C0" stroke="black" stroke-width="1" />
    <circle cx="500" cy="120" r="10" fill="#C0C0C0" stroke="black" stroke-width="1" />
</svg>
```

## 2. "a beacon tower facing the sea"

```xml
<svg width="800" height="600" viewBox="0 0 800 600"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>SVG illustration of a beacon tower facing the sea</title>
    <desc>Generated from prompt: a beacon tower facing the sea</desc>
    
    <!-- Background -->
    <rect width="100%" height="100%" fill="#87CEEB" />

    <!-- Sea -->
    <path d="M0 400 C200 380, 600 420, 800 390 L800 600 L0 600 Z" 
          fill="#006994" />
    
    <!-- Waves -->
    <path d="M0 420 C50 410, 100 430, 150 420 C200 410, 250 430, 300 420 C350 410, 400 430, 450 420 C500 410, 550 430, 600 420 C650 410, 700 430, 750 420 C800 410, 850 430, 900 420" 
          fill="none" stroke="white" stroke-width="2" stroke-opacity="0.3" />
    
    <path d="M0 440 C60 430, 120 450, 180 440 C240 430, 300 450, 360 440 C420 430, 480 450, 540 440 C600 430, 660 450, 720 440 C780 430, 840 450, 900 440" 
          fill="none" stroke="white" stroke-width="1.5" stroke-opacity="0.2" />
    
    <!-- Land/cliff -->
    <path d="M200 400 L400 370 L480 400 L200 400 Z" 
          fill="#8B4513" />
    
    <!-- Lighthouse base -->
    <rect x="300" y="200" width="80" height="170" 
          fill="#F5F5F5" stroke="#333333" stroke-width="2" />
    
    <!-- Lighthouse top -->
    <path d="M290 200 L390 200 L340 150 Z" 
          fill="#CC0000" stroke="#333333" stroke-width="2" />
    
    <!-- Lighthouse light -->
    <circle cx="340" cy="175" r="15" fill="#FFFF00" />
    <circle cx="340" cy="175" r="25" fill="none" stroke="#FFFF00" stroke-width="3" stroke-opacity="0.5" />
    
    <!-- Light beam -->
    <path d="M360 175 L600 100" stroke="#FFFF00" stroke-width="2" stroke-opacity="0.7" />
    <path d="M360 175 L600 120" stroke="#FFFF00" stroke-width="1.5" stroke-opacity="0.5" />
    <path d="M360 175 L600 140" stroke="#FFFF00" stroke-width="1" stroke-opacity="0.3" />
    
    <!-- Windows -->
    <rect x="325" y="220" width="30" height="30" fill="#87CEEB" stroke="#333333" stroke-width="1" />
    <rect x="325" y="270" width="30" height="30" fill="#87CEEB" stroke="#333333" stroke-width="1" />
    <rect x="325" y="320" width="30" height="30" fill="#87CEEB" stroke="#333333" stroke-width="1" />
</svg>
```

## 3. "indigo prisms circling a copper spire"

```xml
<svg width="800" height="600" viewBox="0 0 800 600"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>SVG illustration of indigo prisms circling a copper spire</title>
    <desc>Generated from prompt: indigo prisms circling a copper spire</desc>
    
    <!-- Background -->
    <rect width="100%" height="100%" fill="#F5F5F5" />
    
    <!-- Center spire -->
    <g>
        <path d="M400 150 L450 450 L350 450 Z" fill="#B87333" stroke="#000000" stroke-width="2" />
        <path d="M370 450 L430 450 L430 500 L370 500 Z" fill="#B87333" stroke="#000000" stroke-width="2" />
    </g>
    
    <!-- Circling prisms -->
    <g transform="rotate(0 400 300)">
        <g transform="translate(400 180)">
            <g>
                <polygon points="-30,-15 30,-15 15,15 -15,15" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-15" y1="15" x2="-15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="15" y1="15" x2="15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
    
    <g transform="rotate(60 400 300)">
        <g transform="translate(400 180)">
            <g>
                <polygon points="-30,-15 30,-15 15,15 -15,15" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-15" y1="15" x2="-15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="15" y1="15" x2="15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
    
    <g transform="rotate(120 400 300)">
        <g transform="translate(400 180)">
            <g>
                <polygon points="-30,-15 30,-15 15,15 -15,15" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-15" y1="15" x2="-15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="15" y1="15" x2="15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
    
    <g transform="rotate(180 400 300)">
        <g transform="translate(400 180)">
            <g>
                <polygon points="-30,-15 30,-15 15,15 -15,15" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-15" y1="15" x2="-15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="15" y1="15" x2="15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
    
    <g transform="rotate(240 400 300)">
        <g transform="translate(400 180)">
            <g>
                <polygon points="-30,-15 30,-15 15,15 -15,15" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-15" y1="15" x2="-15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="15" y1="15" x2="15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
    
    <g transform="rotate(300 400 300)">
        <g transform="translate(400 180)">
            <g>
                <polygon points="-30,-15 30,-15 15,15 -15,15" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-15" y1="15" x2="-15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="15" y1="15" x2="15" y2="-15" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
    
    <!-- Inner rotating circle -->
    <g transform="rotate(30 400 300)">
        <g transform="translate(400 250)">
            <g>
                <polygon points="-20,-10 20,-10 10,10 -10,10" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-10" y1="10" x2="-10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="10" y1="10" x2="10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
    
    <g transform="rotate(90 400 300)">
        <g transform="translate(400 250)">
            <g>
                <polygon points="-20,-10 20,-10 10,10 -10,10" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-10" y1="10" x2="-10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="10" y1="10" x2="10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
    
    <g transform="rotate(150 400 300)">
        <g transform="translate(400 250)">
            <g>
                <polygon points="-20,-10 20,-10 10,10 -10,10" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-10" y1="10" x2="-10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="10" y1="10" x2="10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
    
    <g transform="rotate(210 400 300)">
        <g transform="translate(400 250)">
            <g>
                <polygon points="-20,-10 20,-10 10,10 -10,10" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-10" y1="10" x2="-10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="10" y1="10" x2="10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
    
    <g transform="rotate(270 400 300)">
        <g transform="translate(400 250)">
            <g>
                <polygon points="-20,-10 20,-10 10,10 -10,10" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-10" y1="10" x2="-10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="10" y1="10" x2="10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
    
    <g transform="rotate(330 400 300)">
        <g transform="translate(400 250)">
            <g>
                <polygon points="-20,-10 20,-10 10,10 -10,10" fill="#4B0082" stroke="#000000" stroke-width="1.5" />
                <line x1="-10" y1="10" x2="-10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
                <line x1="10" y1="10" x2="10" y2="-10" stroke="#000000" stroke-width="1" stroke-opacity="0.7" />
            </g>
        </g>
    </g>
</svg>
```

## 4. "a violet wood as evening falls"

```xml
<svg width="800" height="600" viewBox="0 0 800 600"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>SVG illustration of a violet wood as evening falls</title>
    <desc>Generated from prompt: a violet wood as evening falls</desc>
    
    <defs>
        <linearGradient id="eveningSky" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#191970" stop-opacity="1" />
            <stop offset="50%" stop-color="#4B0082" stop-opacity="1" />
            <stop offset="100%" stop-color="#800080" stop-opacity="1" />
        </linearGradient>
    </defs>
    
    <!-- Background sky -->
    <rect width="100%" height="100%" fill="url(#eveningSky)" />
    
    <!-- Ground -->
    <path d="M0 450 L800 450 L800 600 L0 600 Z" fill="#4B0050" />
    
    <!-- Distant trees silhouette -->
    <path d="M0 450 C50 430, 100 400, 150 420 C200 390, 250 380, 300 410 C350 370, 400 350, 450 400 C500 360, 550 340, 600 390 C650 370, 700 380, 750 400 C800 390, 850 430, 900 450" 
          fill="#39065A" />
    
    <!-- Mid-distance trees -->
    <!-- Tree 1 -->
    <g transform="translate(100, 400)">
        <rect x="-5" y="0" width="10" height="50" fill="#2A0A3A" />
        <path d="M-30 0 L0 -60 L30 0 Z" fill="#8A2BE2" />
        <path d="M-25 -20 L0 -70 L25 -20 Z" fill="#9400D3" />
        <path d="M-20 -40 L0 -80 L20 -40 Z" fill="#9370DB" />
    </g>
    
    <!-- Tree 2 -->
    <g transform="translate(200, 420)">
        <rect x="-6" y="0" width="12" height="40" fill="#2A0A3A" />
        <path d="M-35 0 L0 -70 L35 0 Z" fill="#8A2BE2" />
        <path d="M-30 -25 L0 -80 L30 -25 Z" fill="#9400D3" />
        <path d="M-25 -50 L0 -90 L25 -50 Z" fill="#9370DB" />
    </g>
    
    <!-- Tree 3 -->
    <g transform="translate(300, 410)">
        <rect x="-4" y="0" width="8" height="45" fill="#2A0A3A" />
        <path d="M-25 0 L0 -50 L25 0 Z" fill="#8A2BE2" />
        <path d="M-20 -15 L0 -60 L20 -15 Z" fill="#9400D3" />
        <path d="M-15 -30 L0 -70 L15 -30 Z" fill="#9370DB" />
    </g>
    
    <!-- Tree 4 -->
    <g transform="translate(420, 430)">
        <rect x="-7" y="0" width="14" height="35" fill="#2A0A3A" />
        <path d="M-40 0 L0 -65 L40 0 Z" fill="#8A2BE2" />
        <path d="M-35 -20 L0 -75 L35 -20 Z" fill="#9400D3" />
        <path d="M-30 -40 L0 -85 L30 -40 Z" fill="#9370DB" />
    </g>
    
    <!-- Tree 5 -->
    <g transform="translate(550, 425)">
        <rect x="-5" y="0" width="10" height="40" fill="#2A0A3A" />
        <path d="M-30 0 L0 -55 L30 0 Z" fill="#8A2BE2" />
        <path d="M-25 -15 L0 -65 L25 -15 Z" fill="#9400D3" />
        <path d="M-20 -35 L0 -75 L20 -35 Z" fill="#9370DB" />
    </g>
    
    <!-- Tree 6 -->
    <g transform="translate(650, 415)">
        <rect x="-6" y="0" width="12" height="45" fill="#2A0A3A" />
        <path d="M-35 0 L0 -60 L35 0 Z" fill="#8A2BE2" />
        <path d="M-30 -20 L0 -70 L30 -20 Z" fill="#9400D3" />
        <path d="M-25 -40 L0 -80 L25 -40 Z" fill="#9370DB" />
    </g>
    
    <!-- Tree 7 -->
    <g transform="translate(750, 405)">
        <rect x="-5" y="0" width="10" height="50" fill="#2A0A3A" />
        <path d="M-30 0 L0 -50 L30 0 Z" fill="#8A2BE2" />
        <path d="M-25 -15 L0 -60 L25 -15 Z" fill="#9400D3" />
        <path d="M-20 -30 L0 -70 L20 -30 Z" fill="#9370DB" />
    </g>
    
    <!-- Foreground trees (bigger and more detailed) -->
    <!-- Foreground Tree 1 -->
    <g transform="translate(170, 460)">
        <rect x="-8" y="0" width="16" height="60" fill="#2A0A3A" />
        <path d="M-45 0 L0 -80 L45 0 Z" fill="#8A2BE2" />
        <path d="M-40 -25 L0 -90 L40 -25 Z" fill="#9400D3" />
        <path d="M-35 -50 L0 -100 L35 -50 Z" fill="#9370DB" />
    </g>
    
    <!-- Foreground Tree 2 -->
    <g transform="translate(370, 480)">
        <rect x="-10" y="0" width="20" height="70" fill="#2A0A3A" />
        <path d="M-50 0 L0 -90 L50 0 Z" fill="#8A2BE2" />
        <path d="M-45 -30 L0 -100 L45 -30 Z" fill="#9400D3" />
        <path d="M-40 -60 L0 -110 L40 -60 Z" fill="#9370DB" />
    </g>
    
    <!-- Foreground Tree 3 -->
    <g transform="translate(600, 470)">
        <rect x="-9" y="0" width="18" height="65" fill="#2A0A3A" />
        <path d="M-48 0 L0 -85 L48 0 Z" fill="#8A2BE2" />
        <path d="M-43 -28 L0 -95 L43 -28 Z" fill="#9400D3" />
        <path d="M-38 -55 L0 -105 L38 -55 Z" fill="#9370DB" />
    </g>
    
    <!-- Evening glow filter -->
    <rect width="100%" height="100%" fill="#800080" fill-opacity="0.1" />
    
    <!-- Stars starting to appear -->
    <circle cx="100" cy="100" r="1" fill="white" fill-opacity="0.7" />
    <circle cx="200" cy="50" r="1" fill="white" fill-opacity="0.6" />
    <circle cx="300" cy="120" r="1" fill="white" fill-opacity="0.8" />
    <circle cx="500" cy="80" r="1" fill="white" fill-opacity="0.7" />
    <circle cx="650" cy="60" r="1" fill="white" fill-opacity="0.6" />
    <circle cx="720" cy="140" r="1" fill="white" fill-opacity="0.8" />
</svg>
```

## 5. "an aubergine satin neckerchief with fringed edges"

```xml
<svg width="800" height="600" viewBox="0 0 800 600"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>SVG illustration of an aubergine satin neckerchief with fringed edges</title>
    <desc>Generated from prompt: an aubergine satin neckerchief with fringed edges</desc>
    
    <defs>
        <pattern id="satinPattern" patternUnits="userSpaceOnUse" width="20" height="20">
            <rect width="20" height="20" fill="#614051"/>
            <path d="M0,0 L20,20 M20,0 L0,20" stroke="#7A5066" stroke-width="0.5" stroke-opacity="0.7" />
        </pattern>
        
        <linearGradient id="satinSheen" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#614051" stop-opacity="1" />
            <stop offset="45%" stop-color="#7A5066" stop-opacity="1" />
            <stop offset="55%" stop-color="#7A5066" stop-opacity="1" />
            <stop offset="100%" stop-color="#614051" stop-opacity="1" />
        </linearGradient>
    </defs>
    
    <!-- Background -->
    <rect width="100%" height="100%" fill="#F8F8F8" />
    
    <!-- Neckerchief - main triangle -->
    <polygon points="250,150 550,150 400,400" 
             fill="url(#satinPattern)" stroke="#4A3040" stroke-width="2" />
    
    <!-- Satin sheen overlay -->
    <polygon points="250,150 550,150 400,400" 
             fill="url(#satinSheen)" fill-opacity="0.3" />
    
    <!-- Fringe edges -->
    <!-- Left edge fringe -->
    <g>
        <line x1="250" y1="150" x2="240" y2="180" stroke="#614051" stroke-width="1.5" />
        <line x1="265" y1="165" x2="255" y2="195" stroke="#614051" stroke-width="1.5" />
        <line x1="280" y1="180" x2="270" y2="210" stroke="#614051" stroke-width="1.5" />
        <line x1="295" y1="195" x2="285" y2="225" stroke="#614051" stroke-width="1.5" />
        <line x1="310" y1="210" x2="300" y2="240" stroke="#614051" stroke-width="1.5" />
        <line x1="325" y1="225" x2="315" y2="255" stroke="#614051" stroke-width="1.5" />
        <line x1="340" y1="240" x2="330" y2="270" stroke="#614051" stroke-width="1.5" />
        <line x1="355" y1="255" x2="345" y2="285" stroke="#614051" stroke-width="1.5" />
        <line x1="370" y1="270" x2="360" y2="300" stroke="#614051" stroke-width="1.5" />
        <line x1="385" y1="285" x2="375" y2="315" stroke="#614051" stroke-width="1.5" />
    </g>
    
    <!-- Right edge fringe -->
    <g>
        <line x1="550" y1="150" x2="560" y2="180" stroke="#614051" stroke-width="1.5" />
        <line x1="535" y1="165" x2="545" y2="195" stroke="#614051" stroke-width="1.5" />
        <line x1="520" y1="180" x2="530" y2="210" stroke="#614051" stroke-width="1.5" />
        <line x1="505" y1="195" x2="515" y2="225" stroke="#614051" stroke-width="1.5" />
        <line x1="490" y1="210" x2="500" y2="240" stroke="#614051" stroke-width="1.5" />
        <line x1="475" y1="225" x2="485" y2="255" stroke="#614051" stroke-width="1.5" />
        <line x1="460" y1="240" x2="470" y2="270" stroke="#614051" stroke-width="1.5" />
        <line x1="445" y1="255" x2="455" y2="285" stroke="#614051" stroke-width="1.5" />
        <line x1="430" y1="270" x2="440" y2="300" stroke="#614051" stroke-width="1.5" />
        <line x1="415" y1="285" x2="425" y2="315" stroke="#614051" stroke-width="1.5" />
    </g>
    
    <!-- Bottom fringe -->
    <g>
        <line x1="390" y1="400" x2="390" y2="430" stroke="#614051" stroke-width="1.5" />
        <line x1="395" y1="400" x2="395" y2="430" stroke="#614051" stroke-width="1.5" />
        <line x1="400" y1="400" x2="400" y2="435" stroke="#614051" stroke-width="1.5" />
        <line x1="405" y1="400" x2="405" y2="430" stroke="#614051" stroke-width="1.5" />
        <line x1="410" y1="400" x2="410" y2="430" stroke="#614051" stroke-width="1.5" />
        
        <line x1="385" y1="400" x2="385" y2="430" stroke="#614051" stroke-width="1.5" />
        <line x1="380" y1="400" x2="380" y2="435" stroke="#614051" stroke-width="1.5" />
        <line x1="375" y1="400" x2="375" y2="430" stroke="#614051" stroke-width="1.5" />
        <line x1="370" y1="400" x2="370" y2="430" stroke="#614051" stroke-width="1.5" />
        
        <line x1="415" y1="400" x2="415" y2="430" stroke="#614051" stroke-width="1.5" />
        <line x1="420" y1="400" x2="420" y2="435" stroke="#614051" stroke-width="1.5" />
        <line x1="425" y1="400" x2="425" y2="430" stroke="#614051" stroke-width="1.5" />
        <line x1="430" y1="400" x2="430" y2="430" stroke="#614051" stroke-width="1.5" />
    </g>
    
    <!-- Fabric folds -->
    <path d="M300,170 Q350,190 350,160" fill="none" stroke="#4A3040" stroke-width="1" stroke-opacity="0.5" />
    <path d="M450,170 Q400,190 400,160" fill="none" stroke="#4A3040" stroke-width="1" stroke-opacity="0.5" />
    <path d="M320,220 Q400,250 480,220" fill="none" stroke="#4A3040" stroke-width="1" stroke-opacity="0.5" />
    <path d="M350,270 Q400,300 450,270" fill="none" stroke="#4A3040" stroke-width="1" stroke-opacity="0.5" />
</svg>
```