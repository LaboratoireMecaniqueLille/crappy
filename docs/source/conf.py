# coding: utf-8

from time import gmtime, strftime
from re import match

__version__ = '2.0.6'

# Project information
# =============================================================================

project = 'Crappy'
author = 'LaMcube and contributors'
copyright = f"{strftime('%Y', gmtime())}, {author}"
version = match(r'\d+\.\d+', __version__).group()
release = __version__

# General configuration
# =============================================================================

# Minimum version requirement for Sphinx
needs_sphinx = ''

# Required extensions for building the documentation
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosectionlabel',
              'sphinx_rtd_theme',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax',
              'sphinx.ext.duration',
              'sphinx_copybutton',
              'sphinx_tabs.tabs',
              'sphinx_toolbox.collapse',
              'sphinx_toolbox.changeset']

# Specific versions required for extensions
needs_extensions = dict()

# URL to cross-reference manpage role
manpages_url = ''

# Specify how to format the current date
today = ''

# Options for figure numbering
# -----------------------------------------------------------------------------

# Automatically number figures, tables, and code-blocks
numfig = False

# The format to use for numbering figures, sections, etc.
numfig_format = {'code-block': 'Listing %s',
                 'figure': 'Fig. %s',
                 'section': 'Section',
                 'table': 'Table %s'}

# Determine if section number is included in figures numbering
numfig_secnum_depth = 1

# Options for highlighting
# -----------------------------------------------------------------------------

# Default language to highlight source code in
highlight_language = 'python3'

# Dictionary of Pygments lexer options
highlight_options = dict()

# The Pygments style to use
pygments_style = 'sphinx'

# Options for HTTP requests
# -----------------------------------------------------------------------------

# Verify server certificates
tls_verify = True

# Path to a directory containing the certificates
tls_cacerts = ''

# The user agent used for HTTP requests
user_agent = ('Mozilla/5.0 (X11; Linux x86_64; rv:100.0) Gecko/'
              '20100101 Firefox/100.0 Sphinx/X.Y.Z')

# Options for internationalisation
# -----------------------------------------------------------------------------

# the language the documents are written in
language = 'en'

# The directories in which to search for additional message catalogs
locale_dirs = ['locales']

# If True, “fuzzy” messages in the message catalogs are used for translation
gettext_allow_fuzzy_translations = False

# use th short or the complete version for text domains
gettext_compact = True

# If True, Sphinx generates UUID information for version tracking in message
# catalogs
gettext_uuid = False

# If True, Sphinx generates location information for messages in message
# catalogs
gettext_location = True

# If True, Sphinx builds a .mo file for each translation catalog file
gettext_auto_build = True

# Enable gettext translation for certain element types
gettext_additional_targets = list()

# The filename format for language-specific figures
figure_language_filename = '{root}.{language}{ext}'

# Control which, if any, classes are added to indicate translation progress
translation_progress_classes = False

# Options for markup
# -----------------------------------------------------------------------------

# The default role for text enclosed between backticks
default_role = None

# Keep warnings as “system message” paragraphs in the rendered documents
keep_warnings = False

# When enabled, emphasise placeholders in option directives
option_emphasise_placeholders = False

# The name of the default domain
primary_domain = 'py'

# A string of reStructuredText that will be included at the end of every
# source file
rst_epilog = ''

# A string of reStructuredText that will be included at the beginning of every
# source file
rst_prolog = ''

# A boolean that decides whether codeauthor and sectionauthor directives
# produce any output
show_authors = False

# Trim spaces before footnote references
trim_footnote_reference_space = False

# Options for Maths
# -----------------------------------------------------------------------------

# A string used for formatting the labels of references to equations
math_eqref_format = '({number})'

# Force all displayed equations to be numbered
math_number_all = False

# If True, displayed math equations are numbered across pages when numfig is
# enabled
math_numfig = True

# A string that defines the separator between section numbers and the equation
# number
math_numsep = '.'

# Options for the nitpicky mode
# -----------------------------------------------------------------------------

# Enables nitpicky mode if True
nitpicky = True

# A set or list of (warning_type, target) tuples that should be ignored when
# generating warnings
nitpick_ignore = {('py:mod', 'smbus2'),
                  ('py:mod', 'Adafruit-Blinka'),
                  ('py:mod', 'adafruit-circuitpython-motorkit')}

# An extended version of nitpick_ignore, which instead interprets the
# warning_type and target strings as regular expressions
nitpick_ignore_regex = set()

# Options for object signatures
# -----------------------------------------------------------------------------

# A boolean that decides whether parentheses are appended to function and
# method role text
add_function_parentheses = True

# If a signature’s length in characters exceeds the number set, each parameter
# within the signature will be displayed on an individual logical line
maximum_signature_line_length = None

# When backslash stripping is enabled then every occurrence of \\ in a domain
# directive will be changed to \, even within string literals
strip_signature_backslash = False

# Create table of contents entries for domain objects
toc_object_entries = True

# A string that determines how domain objects are displayed in their table of
# contents entry
toc_object_entries_show_parents = 'domain'

# Options for source files
# -----------------------------------------------------------------------------

# A list of glob-style patterns that should be excluded when looking for source
# files
exclude_patterns = list()

# A list of glob-style patterns that are used to find source files
include_patterns = ['**']

# This sets the name of the document containing the master toctree directive
master_doc = 'index'

# The file encoding of all source files
source_encoding = 'utf-8-sig'

# A dictionary mapping the file extensions of source files to their file types
source_suffix = {'.rst': 'restructuredtext',
                 '.md': 'markdown'}

# Options for smart quotes
# -----------------------------------------------------------------------------

# If True, the Smart Quotes transform will be used to convert quotation marks
# and dashes to typographically correct entities
smartquotes = True

# Customise the Smart Quotes transform
smartquotes_action = 'qDe'

# Control when the Smart Quotes transform is disabled
smartquotes_excludes = {'languages': ['ja'],
                        'builders': ['man', 'text']}

# Options for templating
# -----------------------------------------------------------------------------

# A string with the fully-qualified name of a callable that returns an instance
# of TemplateBridge
template_bridge = ''

# A list of paths that contain extra templates
templates_path = list()

# Options for warning control
# -----------------------------------------------------------------------------

# Add the type of each warning as a suffix to the warning message
show_warning_types = True

# A list of warning codes to suppress arbitrary warning messages
suppress_warnings = list()

# Builder options
# =============================================================================

# Options for HTML output
# -----------------------------------------------------------------------------

# The theme for HTML output
html_theme = 'sphinx_rtd_theme'

# A dictionary of options that influence the look and feel of the selected
# theme
html_theme_options = {'logo_only': False,
                      'prev_next_buttons_location': 'both',
                      'style_external_links': False,
                      'vcs_pageview_mode': 'blob',
                      'style_nav_header_background': '#2980B9',
                      'flyout_display': 'attached',
                      'version_selector': True,
                      'language_selector': True,
                      # Toc options
                      'collapse_navigation': False,
                      'sticky_navigation': False,
                      'navigation_depth': 4,
                      'includehidden': False,
                      'titles_only': False}

# A list of paths that contain custom themes
html_theme_path = list()

# Stylesheets to use for HTML pages, given by default by the theme
# html_style = list()

# The “title” for HTML documentation generated with Sphinx’s own templates
html_title = f'{project} {release} documentation'

# A shorter “title” for HTML documentation
html_short_title = html_title

# The base URL which points to the root of the HTML documentation
html_baseurl = ''

# The style of line numbers for code-blocks
html_codeblock_linenos_style = 'inline'

# A dictionary of values to pass into the template engine’s context for all
# pages
html_context = {'display_github': True,
                'github_user': 'LaboratoireMecaniqueLille',
                'github_repo': 'crappy',
                'github_version': 'master/docs/source/'}

# If given, this must be the name of an image file that is the logo of the
# documentation
html_logo = ''

# If given, this must be the name of an image file that is the favicon of the
# documentation
html_favicon = ''

# A list of CSS files
html_css_files = tuple()

# A list of JavaScript files
html_js_files = tuple()

# A list of paths that contain custom static files
html_static_path = list()

# A list of paths that contain extra files not directly related to the
# documentation
html_extra_path = list()

# If set, a ‘Last updated on:’ timestamp is inserted into the page footer using
# the given strftime() format
html_last_updated_fmt = '%b %d, %Y'

# Use GMT/UTC (+00:00) instead of the system’s local time zone for the time
# supplied to html_last_updated_fmt
html_last_updated_use_utc = False

# Add link anchors for each heading and description environment
html_permalinks = True

# Text for link anchors for each heading and description environment
html_permalinks_icon = '¶'

# A dictionary defining custom sidebar templates, mapping document names to
# template names
html_sidebars = {'**': ['searchbox.html',
                        'globaltoc.html',
                        'sourcelink.html',
                        'relations.html']}

# Additional templates that should be rendered to HTML pages
html_additional_pages = dict()

# If True, generate domain-specific indices in addition to the general index
html_domain_indices = True

# Controls if an index is added to the HTML documents
html_use_index = True

# Generates two versions of the index: once as a single page with all the
# entries, and once as one page per starting letter
html_split_index = False

# If True, the reStructuredText sources are included in the HTML build as
# _sources/docname
html_copy_source = True

# If True, links to the reStructuredText sources will be added to the sidebar
html_show_sourcelink = True

# The suffix to append to source links, unless files they have this suffix
# already
html_sourcelink_suffix = ''

# If nonempty, an OpenSearch description file will be output
html_use_opensearch = ''

# The file name suffix for generated HTML files
html_file_suffix = '.html'

# The suffix for generated links to HTML files
html_link_suffix = html_file_suffix

# If True, “© Copyright …” is shown in the HTML footer
html_show_copyright = True

# Show a summary of the search result, i.e., the text around the keyword
html_show_search_summary = True

# Add “Created using Sphinx” to the HTML footer
html_show_sphinx = True

# Encoding of HTML output files
html_output_encoding = 'utf-8'

# If True, a list all whose items consist of a single paragraph and/or a
# sub-list all whose items etc… (recursive definition) will not use the <p>
# element for any of its items
html_compact_lists = True

# Suffix for section numbers in HTML output
html_secnumber_suffix = '. '

# Language to be used for generating the HTML full-text search index
html_search_language = language

# A dictionary with options for the search language support
html_search_options = dict()

# The name of a JavaScript file that implements a search results scorer.
html_search_scorer = ''

# Link images that have been resized with a scale option to their original
# full-resolution image
html_scaled_image_link = True

# The maths renderer to use for HTML output
html_math_renderer = 'mathjax'

# Options for Single HTML output
# -----------------------------------------------------------------------------

# A dictionary defining custom sidebar templates
singlehtml_sidebars = html_sidebars

# Options for HTML help output
# -----------------------------------------------------------------------------

# Output file base name for HTML help builder
htmlhelp_basename = f'{project}doc'

# This is the file name suffix for generated HTML help files
htmlhelp_file_suffix = '.html'

# Suffix for generated links to HTML files
htmlhelp_link_suffix = htmlhelp_file_suffix

# Options for Apple Help output
# -----------------------------------------------------------------------------

# The basename for the Apple Help Book
applehelp_bundle_name = project

# The bundle ID for the help book bundle
applehelp_bundle_id = None

# The bundle version, as a string
applehelp_bundle_version = '1'

# The development region
applehelp_dev_region = 'en-us'

# Path to the help bundle icon file or None for no icon
applehelp_icon = None

# The product tag for use with applehelp_kb_url
applehelp_kb_product = f'{project}-{release}'

# The URL for your knowledgebase server
applehelp_kb_url = None

# The URL for remote content
applehelp_remote_url = None

# Tell the help indexer to index anchors in the generated HTML
applehelp_index_anchors = False

# Controls the minimum term length for the help indexer
applehelp_min_term_length = None

# Either a language specification, or the path to a stopwords plist, or None
applehelp_stopwords = language

# Specifies the locale to generate help for
applehelp_locale = language

# Specifies the help book title
applehelp_title = f'{project} Help'

# Specifies the identity to use for code signing
applehelp_codesign_identity = None

# A list of additional arguments to pass to codesign when signing the help book
applehelp_codesign_flags = list()

# The path to the codesign program
applehelp_codesign_path = '/usr/bin/codesign'

# The path to the hiutil program
applehelp_indexer_path = '/usr/bin/hiutil'

# Options for EPUB output
# -----------------------------------------------------------------------------

# The basename for the EPUB file
epub_basename = project

# The HTML theme for the EPUB output
epub_theme = 'epub'

# A dictionary of options that influence the look and feel of the selected
# theme
epub_theme_options = dict()

# The title of the document
epub_title = project

# The description of the document
epub_description = 'unknown'

# The author of the document
epub_author = author

# The name of a person, organisation, etc. that played a secondary role in the
# creation of the content of an EPUB Publication
epub_contributor = 'unknown'

# The language of the document
epub_language = language

# The publisher of the document
epub_publisher = author

# The copyright of the document
epub_copyright = copyright

# An identifier for the document
epub_identifier = 'unknown'

# The publication scheme for the epub_identifier
epub_scheme = 'unknown'

# A unique identifier for the document
epub_uid = 'unknown'

# The cover page information
epub_cover = tuple()

# A list of CSS files
epub_css_files = tuple()

# Metadata for the guide element of content.opf
epub_guide = tuple()

# Additional files that should be inserted before the text generated by Sphinx
epub_pre_files = list()

# Additional files that should be inserted after the text generated by Sphinx
epub_post_files = list()

# A sequence of files that are generated/copied in the build directory but
# should not be included in the EPUB file
epub_exclude_files = list()

# The depth of the table of contents in the file toc.ncx
epub_tocdepth = 3

# This flag determines if a ToC entry is inserted again at the beginning of its
# nested ToC listing
epub_tocdup = True

# This setting control the scope of the EPUB table of contents
epub_tocscope = 'default'

# Try and fix image formats that are not supported by some EPUB readers
epub_fix_images = False

# This option specifies the maximum width of images
epub_max_image_width = 0

# Control how to display URL addresses
epub_show_urls = 'footnote'

# Add an index to the EPUB document
epub_use_index = html_use_index

# It specifies writing direction. It can accept 'horizontal' and 'vertical'
epub_writing_mode = 'horizontal'

# Options for LaTeX output
# -----------------------------------------------------------------------------

# The LaTeX engine to build the documentation
latex_engine = 'pdflatex'

# This value determines how to group the document tree into LaTeX source files
latex_documents = list()

# If given, this must be the name of an image file that is the logo of the
# documentation
latex_logo = ''

# This value determines the topmost sectioning unit
latex_toplevel_sectioning = None

# A list of document names to append as an appendix to all manuals
latex_appendices = list()

# If True, generate domain-specific indices in addition to the general index
latex_domain_indices = True

# Add page references after internal references
latex_show_pagerefs = False

# Control how to display URL addresses
latex_show_urls = 'no'

# Use standard LaTeX’s \multicolumn for merged cells in tables
latex_use_latex_multicolumn = False

# A list of styling classes
latex_table_style = ['booktabs', 'colorrows']

# Use Xindy to prepare the index of general terms
latex_use_xindy = True if latex_engine in {'xelatex', 'lualatex'} else False

# No description available
latex_elements = dict()

# A dictionary mapping 'howto' and 'manual' to names of real document classes
latex_docclass = dict()

# A list of file names to copy to the build directory when building LaTeX
# output
latex_additional_files = list()

# The “theme” that the LaTeX output should use
latex_theme = 'manual'

# A dictionary of options that influence the look and feel of the selected
# theme
latex_theme_options = dict()

# A list of paths that contain custom LaTeX themes as subdirectories
latex_theme_path = list()

# Options for text output
# -----------------------------------------------------------------------------

# Include section numbers in text output
text_add_secnumbers = True

# Determines which end-of-line character(s) are used in text output
text_newlines = 'unix'

# Suffix for section numbers in text output
text_secnumber_suffix = '. '

# A string of 7 characters that should be used for underlining sections
text_sectionchars = '*=-~"+`'

# Options for manual page output
# -----------------------------------------------------------------------------

# This value determines how to group the document tree into manual pages
man_pages = list()

# Add URL addresses after links
man_show_urls = False

# Make a section directory on build man page
man_make_section_directory = True

# Options for Texinfo output
# -----------------------------------------------------------------------------

# This value determines how to group the document tree into Texinfo source
# files
texinfo_documents = list()

# A list of document names to append as an appendix to all manuals
texinfo_appendices = list()

# Generate inline references in a document
texinfo_cross_references = True

# If True, generate domain-specific indices in addition to the general index
texinfo_domain_indices = True

# A dictionary that contains Texinfo snippets that override those that Sphinx
# usually puts into the generated .texi files
texinfo_elements = dict()

# Do not generate a @detailmenu in the “Top” node’s menu containing entries for
# each sub-node in the document
texinfo_no_detailmenu = False

# Control how to display URL addresses
texinfo_show_urls = 'footnote'

# Options for QtHelp output
# -----------------------------------------------------------------------------

# The basename for the qthelp file
qthelp_basename = project

# The namespace for the qthelp file
qthelp_namespace = f'org.sphinx.{project}.{version}'

# The HTML theme for the qthelp output
qthelp_theme = 'nonav'

# A dictionary of options that influence the look and feel of the selected
# theme
qthelp_theme_options = dict()

# Options for XML output
# -----------------------------------------------------------------------------

# Pretty-print the XML
xml_pretty = True

# Options for the linkcheck builder
# -----------------------------------------------------------------------------

# A dictionary that maps a pattern of the source URI to a pattern of the
# canonical URI
linkcheck_allowed_redirects = dict()

# Check the validity of #anchors in links
linkcheck_anchors = True

# A list of regular expressions that match anchors that the linkcheck builder
# should skip when checking the validity of anchors in links
linkcheck_anchors_ignore = ["^!"]

# A list or tuple of regular expressions matching URLs for which the linkcheck
# builder should not check the validity of anchors
linkcheck_anchors_ignore_for_url = tuple()

# A list of regular expressions that match documents in which the linkcheck
# builder should not check the validity of links
linkcheck_exclude_documents = list()

# A list of regular expressions that match URIs that should not be checked when d
# oing a linkcheck build
linkcheck_ignore = list()

# Pass authentication information when doing a linkcheck build
linkcheck_auth = list()

# When a webserver responds with an HTTP 401 (unauthorised) response, the
# current default behaviour of the linkcheck builder is to treat the link as
# “broken”
linkcheck_allow_unauthorized = False

# The linkcheck builder may issue a large number of requests to the same site
# over a short period of time. This setting controls the builder behaviour when
# servers indicate that requests are rate-limited
linkcheck_rate_limit_timeout = 300

# If linkcheck_timeout expires while waiting for a response from a hyperlink,
# the linkcheck builder will report the link as a timeout by default
linkcheck_report_timeouts_as_broken = False

# A dictionary that maps URL (without paths) to HTTP request headers
linkcheck_request_headers = dict()

# The number of times the linkcheck builder will attempt to check a URL before
# declaring it broken
linkcheck_retries = 1

# The duration, in seconds, that the linkcheck builder will wait for a response
# after each hyperlink request
linkcheck_timeout = 30

# The number of worker threads to use when checking links
linkcheck_workers = 5

# Domain options
# =============================================================================

# Options for the C domain
# -----------------------------------------------------------------------------

# A list of identifiers to be recognised as keywords by the C parser
c_extra_keywords = ['alignas', 'alignof', 'bool', 'complex', 'imaginary',
                    'noreturn', 'static_assert', 'thread_local']

# A sequence of strings that the parser should additionally accept as
# attributes
c_id_attributes = tuple()

# If a signature’s length in characters exceeds the number set, each parameter
# within the signature will be displayed on an individual logical line
c_maximum_signature_line_length = None

# A sequence of strings that the parser should additionally accept as
# attributes with one argument
c_paren_attributes = tuple()

# Options for the C++ domain
# -----------------------------------------------------------------------------

# A sequence of strings that the parser should additionally accept as
# attributes
cpp_id_attributes = tuple()

# A list of prefixes that will be ignored when sorting C++ objects in the
# global index
cpp_index_common_prefix = list()

# If a signature’s length in characters exceeds the number set, each parameter
# within the signature will be displayed on an individual logical line
cpp_maximum_signature_line_length = None

# A sequence of strings that the parser should additionally accept as
# attributes with one argument
cpp_paren_attributes = tuple()

# Options for the Javascript domain
# -----------------------------------------------------------------------------

# If a signature’s length in characters exceeds the number set, each parameter
# within the signature will be displayed on an individual logical line
javascript_maximum_signature_line_length = None

# Options for the Python domain
# -----------------------------------------------------------------------------

# A boolean that decides whether module names are prepended to all object names
add_module_names = True

# A list of prefixes that are ignored for sorting the Python module index
modindex_common_prefix = list()

# This value controls how Literal types are displayed
python_display_short_literal_types = True

# If a signature’s length in characters exceeds the number set, each parameter
# within the signature will be displayed on an individual logical line
python_maximum_signature_line_length = None

# Suppress the module name of the python reference if it can be resolved
python_use_unqualified_type_names = False

# Remove doctest flags
trim_doctest_flags = True

# Extension options
# =============================================================================

# sphinx.ext.autodoc
# -----------------------------------------------------------------------------

# This value selects what content will be inserted into the main body of an
# autoclass directive
autoclass_content = 'class'

# This value selects how the signature will be displayed for the class defined
# by autoclass directive
autodoc_class_signature = 'mixed'

# Define the order in which automodule and autoclass members are listed
autodoc_member_order = 'bysource'

# The default options for autodoc directives
autodoc_default_options = {'undoc-members': True}

# Functions imported from C modules cannot be introspected, and therefore the
# signature for such functions cannot be automatically determined. However, it
# is an often-used convention to put the signature into the first line of the
# function’s docstring
autodoc_docstring_signature = True

# This value contains a list of modules to be mocked up
autodoc_mock_imports = list()

# This value controls how to represent typehints
autodoc_typehints = 'signature'

# This value controls whether the types of undocumented parameters and return
# values are documented when autodoc_typehints is set to 'description'
autodoc_typehints_description_target = 'all'

# A dictionary for users defined type aliases that maps a type name to the
# full-qualified object name
autodoc_type_aliases = dict()

# This value controls the format of typehints
autodoc_typehints_format = 'short'

# If True, the default argument values of functions will be not evaluated on
# generating document
autodoc_preserve_defaults = False

# This value controls the behavior of sphinx-build --fail-on-warning during
# importing modules
autodoc_warningiserror = True

# This value controls the docstrings inheritance
autodoc_inherit_docstrings = True

# autodoc supports suppressing warning messages via suppress_warnings
suppress_warnings = list()

# sphinx.ext.intersphinx
# -----------------------------------------------------------------------------

# This config value contains the locations and names of other projects that
# should be linked to in this documentation
intersphinx_mapping = {
  'python': ('https://docs.python.org/3', None),
  'numpy': ('https://numpy.org/doc/stable/', None),
  'matplotlib': ('https://matplotlib.org/stable/', None),
  'psutil': ('https://psutil.readthedocs.io/en/latest/',  None)}

# The maximum number of days to cache remote inventories
intersphinx_cache_limit = 5

# The number of seconds for timeout
intersphinx_timeout = None

# When a non-external cross-reference is being resolved by intersphinx, skip
# resolution if it matches one of the specifications in this list
intersphinx_disabled_reftypes = ['std:doc']

# sphinx.ext.viewcode
# -----------------------------------------------------------------------------

# If this is True, viewcode extension will emit viewcode-follow-imported event
# to resolve the name of the module by other extensions
viewcode_follow_imported_members = True

# If this is True, viewcode extension is also enabled even if you use epub
# builders
viewcode_enable_epub = False

# If set to True, inline line numbers will be added to the highlighted code
viewcode_line_numbers = False

# sphinx.ext.autosectionlabel
# -----------------------------------------------------------------------------

# True to prefix each section label with the name of the document it is in,
# followed by a colon
autosectionlabel_prefix_document = False

# If set, autosectionlabel chooses the sections for labeling by its depth
autosectionlabel_maxdepth = None

# sphinx.ext.napoleon
# -----------------------------------------------------------------------------

# True to parse Google style docstrings
napoleon_google_docstring = True

# True to parse NumPy style docstrings
napoleon_numpy_docstring = False

# True to list __init___ docstrings separately from the class docstring
napoleon_include_init_with_doc = True

# True to include private members with docstrings in the documentation
napoleon_include_private_with_doc = True

# True to include special members with docstrings in the documentation
napoleon_include_special_with_doc = True

# True to use the .. admonition:: directive for the Example and Examples
# sections
napoleon_use_admonition_for_examples = True

# True to use the .. admonition:: directive for Notes sections
napoleon_use_admonition_for_notes = False

# True to use the .. admonition:: directive for References sections
napoleon_use_admonition_for_references = False

# True to use the :ivar: role for instance variables
napoleon_use_ivar = False

# True to use a :param: role for each function parameter
napoleon_use_param = True

# True to use a :keyword: role for each function keyword argument
napoleon_use_keyword = True

# True to use the :rtype: role for the return type
napoleon_use_rtype = False

# True to convert the type definitions in the docstrings as references
napoleon_preprocess_types = False

# A mapping to translate type names to other names or references
napoleon_type_aliases = None

# True to allow using PEP 526 attributes annotations in classes
napoleon_attr_annotations = True

# Add a list of custom sections to include, expanding the list of parsed
# sections
napoleon_custom_sections = list()

# sphinx.ext.mathjax
# -----------------------------------------------------------------------------

# The path to the JavaScript file to include in the HTML files in order to load
# MathJax
# mathjax_path =

# The options to script tag for mathjax
mathjax_options = dict()

# The configuration options for MathJax v3
# mathjax3_config =

# The configuration options for MathJax v2
# mathjax2_config =

# sphinx-copybutton
# -----------------------------------------------------------------------------

# Control which parts of code blocks are skipped when copying
copybutton_exclude = '.linenos'

# Strip input prompt, and only copy lines that begin with a prompt
copybutton_prompt_text = ''

# Indicates whether copybutton_prompt_text should be interpreted as a regular
# expression
copybutton_prompt_is_regexp = False

# Copy only prompt lines, or also output lines
copybutton_only_copy_prompt_lines = False

# Remove the prompt when copying lines beginning with copybutton_prompt_text
copybutton_remove_prompts = False

# Copy or skip empty lines
copybutton_copy_empty_lines = True

# Line continuation character for copying broken lines as only one
copybutton_line_continuation_character = "\\"

# Only applicable to HERE-documents
# copybutton_here_doc_delimiter = "EOT"

# sphinx_tabs.tabs
# -----------------------------------------------------------------------------

# Add builders considered compatible
sphinx_tabs_valid_builders = ['linkcheck']

# Prevent tabs from being closed
sphinx_tabs_disable_tab_closing = True

# Disable CSS loading for tabs
sphinx_tabs_disable_css_loading = False
