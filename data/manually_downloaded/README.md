All the original oral programs for past MICCAI conferences are here:

- 2021, pages 1-6: <https://www.miccai2021.org/files/downloads/MICCAI2021%20-%20Oral-Presentation-Schedule.pdf>
- 2022, pages 32-39: <https://conferences.miccai.org/2022/files/downloads/MICCAI2022-Program-Book.pdf>
- 2023, pages 30-41: <https://conferences.miccai.org/2023/files/downloads/MICCAI2023-Program-Book.pdf>
- 2024, pages 42-49: <https://conferences.miccai.org/2024/files/downloads/MICCAI2024-Program-Book.pdf>
- 2025, pages 44-53: <https://conferences.miccai.org/2025/files/downloads/MICCAI2025-Program-Book.pdf>

So, I downloaded them manually in `ProgramBooks/`, and then ran the following commands to extract only the relevant pages:

```bash
cp ProgramBooks/MICCAI2021\ -\ Oral-Presentation-Schedule.pdf OralSchedules/2021.pdf
pdftk ProgramBooks/MICCAI2022-Program-Book.pdf cat 32-39 output OralSchedules/2022.pdf
pdftk ProgramBooks/MICCAI2023-Program-Book.pdf cat 30-41 output OralSchedules/2023.pdf
pdftk ProgramBooks/MICCAI2024-Program-Book.pdf cat 42-49 output OralSchedules/2024.pdf
pdftk ProgramBooks/MICCAI2025-Program-Book.pdf cat 44-53 output OralSchedules/2025.pdf
```
