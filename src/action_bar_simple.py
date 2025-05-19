"""
Simple Action Bar Module for Streamlit

This module provides CSS and HTML components for creating a fixed action bar
that stays at the bottom of the screen in Streamlit applications.
"""

def get_action_bar_css():
    """
    Return CSS for styling the action bar.
    
    Returns:
        str: CSS styling as a string
    """
    return """
    <style>
    /* Action bar container */
    .action-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 100;
        background-color: rgba(255, 255, 255, 0.95);
        box-shadow: 0 -4px 10px rgba(0, 0, 0, 0.1);
        padding: 10px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        backdrop-filter: blur(10px);
        border-top: 1px solid #e0e0e0;
    }
    
    /* Left section for status/info */
    .action-bar-status {
        flex: 1;
        display: flex;
        align-items: center;
    }
    
    /* Center section for primary actions */
    .action-bar-actions {
        flex: 2;
        display: flex;
        justify-content: center;
        gap: 10px;
    }
    
    /* Right section for secondary actions */
    .action-bar-aux {
        flex: 1;
        display: flex;
        justify-content: flex-end;
        gap: 8px;
    }
    
    /* General button styling */
    .action-bar-button {
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
    }
    
    /* Primary action button */
    .action-primary {
        background-color: #4CAF50;
        color: white;
        border: none;
    }
    .action-primary:hover {
        background-color: #45a049;
    }
    
    /* Secondary action button */
    .action-secondary {
        background-color: #f8f9fa;
        color: #212529;
        border: 1px solid #dee2e6;
    }
    .action-secondary:hover {
        background-color: #e9ecef;
    }
    
    /* Danger action button */
    .action-danger {
        background-color: #dc3545;
        color: white;
        border: none;
    }
    .action-danger:hover {
        background-color: #c82333;
    }
    
    /* Button icon spacing */
    .button-icon {
        margin-right: 6px;
    }
    
    /* Status indicator styles */
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-success {
        background-color: #28a745;
    }
    .status-warning {
        background-color: #ffc107;
    }
    .status-error {
        background-color: #dc3545;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .action-bar {
            flex-direction: column;
            padding: 10px;
        }
        .action-bar-status, .action-bar-actions, .action-bar-aux {
            width: 100%;
            justify-content: center;
            margin: 5px 0;
        }
    }
    </style>
    """

def get_simple_bar_html(status_text="Ready", 
                        primary_button_text="Apply Changes", 
                        primary_onclick="", 
                        secondary_button_text="Reset", 
                        secondary_onclick="",
                        show_feedback=False):
    """
    Generate HTML for a simple action bar.
    
    Args:
        status_text (str): Text to display in the status area
        primary_button_text (str): Text for the primary action button
        primary_onclick (str): JavaScript onclick handler for primary button
        secondary_button_text (str): Text for the secondary action button
        secondary_onclick (str): JavaScript onclick handler for secondary button
        show_feedback (bool): Whether to show feedback buttons
        
    Returns:
        str: HTML for the action bar
    """
    # Base HTML structure
    html = f"""
    <div class="action-bar">
        <div class="action-bar-status">
            <div class="status-indicator status-success"></div>
            <span>{status_text}</span>
        </div>
        <div class="action-bar-actions">
            <button class="action-bar-button action-primary" onclick="{primary_onclick}">
                {primary_button_text}
            </button>
            <button class="action-bar-button action-secondary" onclick="{secondary_onclick}">
                {secondary_button_text}
            </button>
        </div>
    """
    
    # Add feedback buttons if requested
    if show_feedback:
        html += """
        <div class="action-bar-aux">
            <button class="action-bar-button action-secondary" title="Was this helpful?">
                <span class="button-icon">👍</span>
            </button>
            <button class="action-bar-button action-secondary" title="Need improvements?">
                <span class="button-icon">👎</span>
            </button>
        </div>
        """
    else:
        html += '<div class="action-bar-aux"></div>'
    
    # Close the main container
    html += "</div>"
    
    return html
