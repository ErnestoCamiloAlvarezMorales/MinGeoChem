# MinGeoChem
MinGeoChem: An interactive web application for optical mineralogy, 3D crystallography, and thermodynamic phase diagram modeling (Clapeyron/Gibbs). Built with Python &amp; Streamlit.

🌋 MinGeoChem
An interactive web application designed for geoscience students and professionals to explore optical mineralogy, crystallography, and thermodynamic phase diagrams. Built with Python and Streamlit.


🚀 Features


📖 Optical Atlas: Browse a database of rock-forming minerals with high-resolution microprobe images (PPL & XPL) and a dynamically parsed table of their optical properties.

📐 3D Crystal Symmetry: Interactive 3D visualizer of the 7 crystal systems. Wireframe unit cells are generated mathematically based on axial lengths and interaxial angles.

🧪 Thermodynamic Calculator: Calculate enthalpy (ΔH), entropy (ΔS), and Gibbs free energy (ΔG) for mineral reactions using Hess's Law and numerical integration of Maier-Kelly heat capacities (Cp). Includes a Gibbs Phase Rule calculator.

🎮 Mineralogical Quiz: Test your mineral identification skills using microprobe images. Features a scoring system to track progress.

🌋 3D Phase Diagrams: Generate Pressure-Temperature-Depth (P-T-D) phase diagrams using the Clapeyron equation. Visualize univariant curves and calculate the triple point of polymorphic phases (e.g., SiO2 system) in an interactive 3D space.

⚙️ Custom Data Upload: Upload your own CSV files for mineralogy and geochemistry to use the app with your custom datasets.


🛠 Tech Stack


Frontend/UI: Streamlit

Data Handling: Pandas, NumPy

Thermodynamic Math: SciPy (Numerical Integration)

3D Visualization: Plotly


📦 Data Sources


Optical Mineralogy: Web-scraped and curated from the University of Granada Mineralogy Database.

Thermodynamics: Standard state properties and Maier-Kelly heat capacity coefficients based on published scientific literature.


💻 Installation & Local Usage


Clone the repository:git clone https://github.com/your-username/MinGeoChem.gitcd MinGeoChem
