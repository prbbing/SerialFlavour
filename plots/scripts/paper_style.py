"""Keep plot encodings; move descriptive titles and notes into paper captions."""
import re


def simplify(fig, name):
    from pipeline_common import figure_identity
    name = figure_identity(name)+'.png'
    removed = []
    for text in list(fig.texts):
        if text.get_text():
            removed.append(text.get_text())
        text.set_visible(False)
        text.set_in_layout(False)
    for legend in [*fig.legends, *[ax.get_legend() for ax in fig.axes if ax.get_legend()]]:
        legend.get_title().set_text('')
        for text in legend.get_texts():
            before = text.get_text()
            after = re.sub(r'\s*\(n\s*=\s*\d+\)', '', before)
            after = re.sub(r':\s*(solid|dashed|dotted)', '', after)
            text.set_text(after)
            if before != after:
                removed.append(before)
    if 'rejection_heatmap' in name:
        fig.subplots_adjust(top=.87)
        fig.axes[-1].set_position([.925, .14, .014, .73])
    elif name.startswith('p122_'):
        # Nested ratio panels share the outer GridSpec.
        grid = fig.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec()
        grid.update(top=.91, bottom=.06)
        for legend in fig.legends:
            legend.set_bbox_to_anchor((.5, .995))
        fig.set_size_inches(13.5, 11.8)
    elif 'with_upstream_fit_overlay' in name and 'six_working_points' in name:
        fig.set_size_inches(18, 13.5)
        fig.subplots_adjust(top=.90, bottom=.08, hspace=.35)
        for legend, y in zip(fig.legends, [.995, .96]):
            legend.set_bbox_to_anchor((.5, y))
    elif 'with_upstream_fit_overlay' in name:
        fig.set_size_inches(18, 6.7)
        fig.subplots_adjust(top=.81, bottom=.17)
        for legend, y in zip(fig.legends, [.995, .93]):
            legend.set_bbox_to_anchor((.5, y))
    elif 'absolute_rejection_sorted' in name:
        fig.set_size_inches(14.4, 6.1)
        fig.subplots_adjust(top=.84, bottom=.23)
        for legend in fig.legends:
            legend.set_bbox_to_anchor((.5, .995))
    elif 'relative_rejection_vs_data_amount' in name:
        fig.set_size_inches(18, 4.9)
        fig.subplots_adjust(top=.82, bottom=.14)
        for legend in fig.legends:
            legend.set_bbox_to_anchor((.5, .995))
    return removed


def publication_style(fig, name):
    from matplotlib.text import Text
    from pipeline_common import display_name
    for text in fig.findobj(match=Text):
        text.set_text(display_name(text.get_text()))
    removed = simplify(fig, name)
    assert all(display_name(t.get_text()) == t.get_text()
               for t in fig.findobj(match=Text) if t.get_visible())
    return removed
