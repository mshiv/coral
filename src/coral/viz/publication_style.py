"""Caption-first presentation, without changing plotted data or metrics."""


def caption_first(fig, axes):
    for text in list(fig.texts):
        text.remove()
    for i, ax in enumerate(axes):
        ax.set_title('')
        ax.text(.02,.98,f'({chr(97+i)})',transform=ax.transAxes,
                va='top',ha='left',weight='bold',fontsize=12,
                bbox=dict(fc='white',ec='none',alpha=.8,pad=1))
    fig.tight_layout()
