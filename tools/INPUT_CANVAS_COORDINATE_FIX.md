# Input Canvas Coordinate System Fix

## Problem
The original input canvas used the default HTML Canvas coordinate system:
- **Origin**: Top-left corner (0, 0)
- **X-axis**: Points right (positive direction)
- **Y-axis**: Points down (positive direction)

This caused a mismatch with mathematical coordinate systems where Y typically points upward, making it difficult to align the input geometry with the normalized output.

## Solution
The input canvas now uses a **mathematical coordinate system**:
- **Origin**: Bottom-left corner (0, 0)
- **X-axis**: Points right (positive direction) ✓
- **Y-axis**: Points up (positive direction) ✓

## Implementation Details

### 1. Point Input Transformation (`addPoint`)
When the user clicks on the canvas, the click coordinates are transformed:

```javascript
// Get raw canvas click position (top-left origin)
const canvasX = (event.clientX - rect.left) * scaleX;
const canvasY = (event.clientY - rect.top) * scaleY;

// Transform to mathematical coordinates (bottom-left origin, Y up)
const x = canvasX;                           // X unchanged
const y = this.inputCanvas.height - canvasY; // Y inverted
```

### 2. Point Storage
Points are now stored in **mathematical coordinates**:
- `point.x`: Distance from left edge
- `point.y`: Distance from bottom edge (0 at bottom, increases upward)

### 3. Canvas Rendering (`drawInputCanvas`)
When rendering, mathematical coordinates are converted back to canvas coordinates:

```javascript
// Helper function to convert math Y to canvas Y
const toCanvasY = (mathY) => canvas.height - mathY;

// When drawing:
ctx.arc(point.x, toCanvasY(point.y), radius, 0, 2 * Math.PI);
```

### 4. Visual Indicators
Added coordinate axes to the input canvas:
- **X-axis**: Horizontal line at the bottom with arrow pointing right
- **Y-axis**: Vertical line on the left with arrow pointing up
- **Labels**: "0" at origin, "+X" and "+Y" labels
- **Style**: Dashed lines to distinguish from user geometry

## Code Changes

### Modified Functions

1. **`addPoint(event)`**
   - Transforms click coordinates from canvas space to mathematical space
   - Stores points with Y-up coordinate system

2. **`drawInputCanvas()`**
   - Added `toCanvasY()` helper function
   - Calls `drawInputCoordinateAxes()` to draw reference axes
   - Converts all Y coordinates when rendering points and lines

3. **`drawInputCoordinateAxes(ctx, canvas, toCanvasY)`** (NEW)
   - Draws dashed X and Y axes
   - Adds arrows to indicate positive directions
   - Labels the origin and axis directions

## Visual Features

### Coordinate Axes Display
- **Dashed lines**: To distinguish from geometry
- **Arrows**: Show positive direction of each axis
- **Origin marker**: "0" label at bottom-left
- **Axis labels**: "+X" (right) and "+Y" (up)
- **Margins**: 30px from canvas edges

### User Experience
- User sees familiar mathematical coordinate system
- Origin clearly marked at bottom-left
- Y coordinates increase as you move up the canvas
- X coordinates increase as you move right
- Axes are visible even when no points are added

## Benefits

1. **Consistency**: Input coordinates now match mathematical conventions
2. **Predictability**: Y-up matches normalized output coordinate interpretation
3. **Clarity**: Visual axes make the coordinate system obvious
4. **Compatibility**: Easier to compare input and output geometries

## Example

If a user clicks at the **visual center** of a 500×400 canvas:

### Old System (Top-left origin)
- Click position: (250, 200)
- Stored as: `{x: 250, y: 200}`
- Y value near middle of screen

### New System (Bottom-left origin)
- Click position: (250, 200) canvas coordinates
- Stored as: `{x: 250, y: 200}` mathematical coordinates
  - Where y=200 means 200 pixels **up from bottom**
- Y value correctly represents upward distance

## API Interaction

The coordinates sent to the API are now in mathematical coordinate system format:
```javascript
const coordinates = this.points.map(point => [point.x, point.y]);
// Example: [[250, 200], [100, 350], ...] 
// where Y values increase upward from bottom
```

This ensures better alignment between input geometry and normalized output, especially when comparing polar and Cartesian representations.

## Testing

To verify the coordinate system:
1. Click at the bottom-left corner → should show coordinates near (0, 0)
2. Click at the bottom-right corner → should show coordinates near (500, 0)
3. Click at the top-left corner → should show coordinates near (0, 400)
4. Click at the top-right corner → should show coordinates near (500, 400)
5. Y values should **increase** as you click higher on the canvas

## UI Updates

Added coordinate system hint in HTML:
```html
<p class="text-xs text-gray-500 mb-2">
  📐 Coordinate System: Origin at bottom-left, +X right, +Y up
</p>
```

This provides immediate visual feedback about the coordinate system being used.
