window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

// Teaser magnifier lens
function initTeaserMagnifier() {
    const figure = document.querySelector('.teaser-figure');
    if (!figure) return;

    const img = figure.querySelector('.teaser-image');
    const lens = figure.querySelector('.teaser-lens');
    if (!img || !lens) return;

    const zoom = 2.4;

    function updateLensBackground() {
        const rect = img.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        lens.style.backgroundImage = `url('${img.currentSrc || img.src}')`;
        lens.style.backgroundSize = `${rect.width * zoom}px ${rect.height * zoom}px`;
    }

    function moveLens(event) {
        const rect = img.getBoundingClientRect();
        const lensRect = lens.getBoundingClientRect();
        const lensW = lensRect.width || lens.offsetWidth;
        const lensH = lensRect.height || lens.offsetHeight;

        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
            lens.style.display = 'none';
            return;
        }

        lens.style.display = 'block';

        const left = Math.min(Math.max(x - lensW / 2, 0), rect.width - lensW);
        const top = Math.min(Math.max(y - lensH / 2, 0), rect.height - lensH);

        lens.style.left = `${left}px`;
        lens.style.top = `${top}px`;

        const bgX = x * zoom - lensW / 2;
        const bgY = y * zoom - lensH / 2;
        lens.style.backgroundPosition = `-${bgX}px -${bgY}px`;
    }

    function hideLens() {
        lens.style.display = 'none';
    }

    if (img.complete) {
        updateLensBackground();
    } else {
        img.addEventListener('load', updateLensBackground);
    }

    window.addEventListener('resize', updateLensBackground);
    figure.addEventListener('mousemove', moveLens);
    figure.addEventListener('mouseleave', hideLens);
}

// X0 row overlay reveal slider
function initX0Overlay() {
    const slider = document.getElementById('x0Overlay');
    const row = document.querySelector('.artifact-x0');
    if (!slider || !row) return;

    function update() {
        const val = Number(slider.value);
        const t = Math.max(0, Math.min(1, val / 100));
        row.style.setProperty('--overlay-alpha', t.toFixed(2));
        row.classList.toggle('is-active', val > 0);
    }

    slider.addEventListener('input', update);
    slider.value = 0;
    update();
}

// XT row overlay reveal slider
function initXtOverlay() {
    const slider = document.getElementById('xtOverlay');
    const row = document.querySelector('.artifact-xt');
    if (!slider || !row) return;

    function update() {
        const val = Number(slider.value);
        const t = Math.max(0, Math.min(1, val / 100));
        row.style.setProperty('--overlay-alpha', t.toFixed(2));
        row.classList.toggle('is-active', val > 0);
    }

    slider.addEventListener('input', update);
    slider.value = 0;
    update();
}

// Reveal animation for artifact section
function initArtifactReveal() {
    const targets = document.querySelectorAll(
        '.artifact-check .artifact-title, .artifact-check .artifact-description, .artifact-check .artifact-row'
    );
    if (targets.length === 0) return;

    targets.forEach(el => el.classList.add('artifact-animate'));

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('in-view');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.2 });

    targets.forEach(el => observer.observe(el));
}

// Reveal animation for qualitative results
function initResultsReveal() {
    const pairs = document.querySelectorAll('.results-pair');
    if (pairs.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('in-view');
                setTimeout(() => {
                    entry.target.classList.add('settled');
                }, 1900);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.2 });

    pairs.forEach(pair => observer.observe(pair));
}

// Reveal animation for dataset section
function initDatasetReveal() {
    const section = document.querySelector('.dataset-results');
    if (!section) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('in-view');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.2 });

    observer.observe(section);
}

// Toggle zoom images in qualitative results
function initResultsZoom() {
    const buttons = document.querySelectorAll('.results-zoom-btn');
    if (buttons.length === 0) return;

    buttons.forEach(button => {
        const item = button.closest('.results-item');
        const img = item ? item.querySelector('.results-image') : null;
        if (!img) return;

        button.addEventListener('click', () => {
            const isActive = button.getAttribute('aria-pressed') === 'true';
            const nextSrc = isActive ? img.dataset.src : img.dataset.zoom;
            if (nextSrc) {
                img.src = nextSrc;
            }
            button.setAttribute('aria-pressed', String(!isActive));
            button.textContent = isActive ? 'Show zoom' : 'Show original';
        });
    });
}

// Overview of DIAMOND reveal
function initPipelineReveal() {
    const card = document.querySelector('.pipeline-card');
    if (!card) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('in-view');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.2 });

    observer.observe(card);
}

// Read more toggle for pipeline overview
function initPipelineToggle() {
    const card = document.querySelector('.pipeline-card');
    const button = document.querySelector('.pipeline-toggle');
    if (!card || !button) return;

    button.addEventListener('click', () => {
        const isOpen = card.classList.toggle('is-open');
        button.setAttribute('aria-expanded', String(isOpen));
        button.textContent = isOpen ? 'Read less' : 'Read more';
    });
}

// Dataset results auto-toggle
function initDatasetResults() {
    const section = document.querySelector('.dataset-results');
    if (!section) return;

    const tabButtons = Array.from(section.querySelectorAll('.dataset-tab'));
    const cards = Array.from(section.querySelectorAll('.dataset-card'));
    if (tabButtons.length === 0 || cards.length === 0) return;

    const datasetPairs = {
        people: [[38, 39], [40, 41], [42, 43], [44, 45]],
        words: [[30, 31], [32, 33], [34, 35], [36, 37]],
        animals: [[46, 47], [48, 49], [50, 51], [52, 53]]
    };

    const imageItems = cards.map(card => {
        const media = card.querySelector('.dataset-media');
        const baselineImg = card.querySelector('.dataset-image.baseline');
        const diamondImg = card.querySelector('.dataset-image.diamond');
        if (!media || !baselineImg || !diamondImg) return null;
        return { media, baselineImg, diamondImg, showingDiamond: false };
    });

    let autoTimer = null;
    const switchDelay = 3200;

    const toggleImages = () => {
        imageItems.forEach(item => {
            if (!item) return;
            item.showingDiamond = !item.showingDiamond;
            item.media.classList.toggle('show-diamond', item.showingDiamond);
        });
    };

    const startAuto = () => {
        if (autoTimer) return;
        autoTimer = window.setInterval(toggleImages, switchDelay);
    };

    const stopAuto = () => {
        if (autoTimer) {
            window.clearInterval(autoTimer);
            autoTimer = null;
        }
    };

    function setDataset(name) {
        const pairs = datasetPairs[name];
        if (!pairs) return;

        tabButtons.forEach(button => {
            const isActive = button.dataset.dataset === name;
            button.classList.toggle('is-active', isActive);
            button.setAttribute('aria-selected', isActive ? 'true' : 'false');
        });

        cards.forEach((card, index) => {
            const pair = pairs[index];
            const item = imageItems[index];
            const baselineImg = item ? item.baselineImg : card.querySelector('.dataset-image.baseline');
            const diamondImg = item ? item.diamondImg : card.querySelector('.dataset-image.diamond');
            if (!pair || !baselineImg || !diamondImg) return;

            const baseline = `static/images/dataset/${name}/${pair[0]}.png`;
            const diamond = `static/images/dataset/${name}/${pair[1]}.png`;

            const altBase = `${name} example ${index + 1}`;
            baselineImg.src = baseline;
            baselineImg.alt = `${altBase} baseline`;
            diamondImg.src = diamond;
            diamondImg.alt = `${altBase} diamond`;
            if (item) {
                item.showingDiamond = false;
                item.media.classList.remove('show-diamond');
            }
        });
    }

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            stopAuto();
            setDataset(button.dataset.dataset);
            startAuto();
        });
    });

    const initial = tabButtons.find(button => button.classList.contains('is-active'))?.dataset.dataset || 'people';
    setDataset(initial);
    startAuto();
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();
    
    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

    // Setup teaser magnifier
    initTeaserMagnifier();

    // Setup x0 overlay slider
    initX0Overlay();

    // Setup xt overlay slider
    initXtOverlay();

    // Setup artifact reveal animation
    initArtifactReveal();

    // Setup results reveal animation
    initResultsReveal();

    // Setup results zoom toggle
    initResultsZoom();

    // Setup pipeline reveal
    initPipelineReveal();
    initPipelineToggle();

    // Setup dataset results auto-toggle
    initDatasetResults();
    initDatasetReveal();

})
