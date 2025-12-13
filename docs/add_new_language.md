# Adding a New Language

This guide describes how to add support for a new language in the application. The process involves two parts: the backend (Flask) and the frontend (Vue.js).

## 1. Backend (Flask-Babel)

The backend handles translations for server-side messages (e.g., API errors, logs).

**WorkingDirectory:** `web_app/`

### Prerequisites
Ensure you have the development dependencies installed, including `Babel`.

### Steps:

1.  **Extract Messages** (if you have added new translatable strings in the Python code):
    ```bash
    pybabel extract -F babel.cfg -k _ -o messages.pot .
    ```

2.  **Initialize a New Language**:
    Replace `<language_code>` with the ISO 639-1 code (e.g., `es` for Spanish, `fr` for French).
    ```bash
    pybabel init -i messages.pot -d translations -l <language_code>
    ```

3.  **Translate Strings**:
    -   Navigate to `web_app/translations/<language_code>/LC_MESSAGES/`.
    -   Open `messages.po` in a text editor or a translation tool (like Poedit).
    -   Add translations for each `msgid`.

4.  **Compile Translations**:
    This converts the `.po` files into binary `.mo` files used by the application.
    ```bash
    pybabel compile -d translations
    ```

5.  **Update Existing Languages** (if you extracted new messages):
    ```bash
    pybabel update -i messages.pot -d translations
    ```

---

## 2. Frontend (Vue-i18n)

The frontend handles translations for the user interface.

**WorkingDirectory:** `frontend/`

### Steps:

1.  **Create a Locale File**:
    -   Create a new JSON file in `frontend/src/locales/` named `<language_code>.json` (e.g., `es.json`).
    -   Copy the structure from `en.json` and translate the values.

2.  **Register the Language**:
    -   Open `frontend/src/i18n.js`.
    -   Import your new locale file:
        ```javascript
        import es from './locales/es.json'
        ```
    -   Add it to the `messages` object:
        ```javascript
        const i18n = createI18n({
            // ...
            messages: {
                en,
                de,
                es // Add this line
            }
        })
        ```

3.  **Add to UI Selector**:
    -   Open `frontend/src/components/SettingsModal.vue`.
    -   Find the language selection section (search for `changeLocale`).
    -   Add a new button for the language:
        ```html
        <button
            @click="changeLocale('es')"
            :class="[
                locale === 'es' ? 'bg-brand-primary ...' : 'bg-white/5 ...',
                'px-4 py-2 rounded-md text-sm font-medium transition-all duration-200'
            ]"
        >
            Español
        </button>
        ```
